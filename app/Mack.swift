// Mack: menu-bar app for Mack's background tasks.
//
// Mack takes on big jobs (a long report, a small codebase) and works through them
// slowly in the background. The menu-bar icon shows progress, "New Task…" starts
// one, and the dashboard shows the work as it is written. The app starts Mack's
// local engine and stops it on quit; unfinished tasks resume on the next launch.
//
// The earlier explainer is still in the menu: Control-Option-M, drag a box over
// text on screen, and Mack explains it in a floating panel. Text is recognized
// on-device with Vision and sent only to the engine on 127.0.0.1.
//
// Build with app/build.sh. Hidden test modes (no UI permissions needed):
//   Mack --selftest               render text to an image and OCR it back
//   Mack --ocr <image>            print the text recognized in an image file
//   Mack --render-test <out.png>  draw the panel with sample content to a PNG
//   Mack --explain-text <text>    send text to the engine and print the streamed reply
//   Mack --pdf-text <pdf>         text layer of a PDF, or OCR of its first 3 pages if it has none
//   Mack --make-image <out.png> <text>      fixtures: text rendered to an image
//   Mack --make-pdf <out.pdf> <text> [--scan]  fixtures: a PDF with a text layer, or image-only with --scan
//   Mack --diagnose <out.txt>     write the screen-access diagnosis (counts only) to a file;
//                                 run it through `open -n Mack.app --args --diagnose <file>`
//                                 so it executes under the app's own identity

import AppKit
import Carbon.HIToolbox
import PDFKit
import SwiftUI
import Vision

// MACK_URL overrides the engine address (used by the test modes below).
let serverURL = URL(string: ProcessInfo.processInfo.environment["MACK_URL"] ?? "http://127.0.0.1:8082")!

// MARK: - OCR

func recognizeText(in image: CGImage) -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.automaticallyDetectsLanguage = true
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    guard (try? handler.perform([request])) != nil, let results = request.results else { return "" }
    // Vision's origin is bottom-left. Read top to bottom; boxes on roughly the same
    // line are joined left to right so table rows and wrapped UI text stay together.
    let items = results.compactMap { obs -> (CGRect, String)? in
        guard let best = obs.topCandidates(1).first else { return nil }
        return (obs.boundingBox, best.string)
    }.sorted { $0.0.midY > $1.0.midY }
    var lines: [[(CGRect, String)]] = []
    for item in items {
        if let last = lines.last?.last, abs(last.0.midY - item.0.midY) < min(last.0.height, item.0.height) * 0.6 {
            lines[lines.count - 1].append(item)
        } else {
            lines.append([item])
        }
    }
    return lines.map { $0.sorted { $0.0.minX < $1.0.minX }.map { $0.1 }.joined(separator: "  ") }
        .joined(separator: "\n")
}

func loadCGImage(_ url: URL) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}

// MARK: - Screen access

/// Whether this process can really see other apps' windows. macOS only reveals
/// window titles to processes with Screen Recording access, so this is the ground
/// truth; CGPreflightScreenCaptureAccess can say yes while captures come back with
/// nothing but the desktop picture (for example after the app was rebuilt).
func canSeeOtherWindows() -> Bool {
    guard let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
            as? [[String: Any]] else { return false }
    let me = getpid()
    var normalWindows = 0
    for w in windows {
        guard let pid = w[kCGWindowOwnerPID as String] as? pid_t, pid != me,
              (w[kCGWindowLayer as String] as? Int) == 0 else { continue }
        normalWindows += 1
        if let name = w[kCGWindowName as String] as? String, !name.isEmpty { return true }
    }
    return normalWindows == 0  // nothing open to judge by: do not block the capture
}

struct ScreenDiagnosis {
    var preflight = false
    var seesWindows = false
    var captureStatus: Int32 = -1
    var width = 0
    var height = 0
    var characters = 0
    var lines = 0

    /// Capture the main display non-interactively and count the text in it. Only
    /// counts are kept; the image is deleted and its text is not stored or shown.
    static func run() -> ScreenDiagnosis {
        var d = ScreenDiagnosis()
        d.preflight = CGPreflightScreenCaptureAccess()
        d.seesWindows = canSeeOtherWindows()
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("mack-diag-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: file) }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        task.arguments = ["-x", "-m", "-t", "png", file.path]
        guard (try? task.run()) != nil else { return d }
        task.waitUntilExit()
        d.captureStatus = task.terminationStatus
        if let image = loadCGImage(file) {
            d.width = image.width
            d.height = image.height
            let text = recognizeText(in: image)
            d.characters = text.count
            d.lines = text.isEmpty ? 0 : text.components(separatedBy: "\n").count
        }
        return d
    }

    var works: Bool { characters >= 40 && seesWindows }

    var report: String {
        """
        permission reported by macOS: \(preflight ? "yes" : "no")
        can see other apps' windows: \(seesWindows ? "yes" : "no")
        test capture: \(width == 0 ? "failed (status \(captureStatus))" : "\(width)×\(height) px")
        text found in it: \(characters) characters on \(lines) lines
        """
    }
}

let permissionSteps = """
    Open System Settings → Privacy & Security → Screen & System Audio Recording.
    If Mack is in the list, turn it off, select it and remove it with the minus button. Then quit Mack, open it again, click the icon, and allow it when macOS asks.

    Until then you can copy some text and choose “Explain Copied Text” from Mack's menu (right-click the icon).
    """

// MARK: - Model shared by the panel

enum Style: String { case explain, simpler, detail, summary }

final class ExplainModel: ObservableObject {
    @Published var status = ""
    @Published var answer = ""
    @Published var captured = ""
    @Published var busy = false
    @Published var notice: String? = nil  // shown instead of an answer (errors, setup steps)
    @Published var readSummary = ""  // what Mack read off the screen, shown under the header
    @Published var showCaptured = false
    @Published var question = ""
    @Published var meta = ""

    var onStyle: (Style) -> Void = { _ in }
    var onAsk: (String) -> Void = { _ in }
    var onClose: () -> Void = {}
    var onRecapture: () -> Void = {}
}

/// The model writes Markdown. Inline styles are rendered; block syntax is turned
/// into plain typographic equivalents, which reads cleanly while text streams in.
func renderMarkdown(_ text: String) -> AttributedString {
    var out = AttributedString()
    let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    let lines = text.components(separatedBy: "\n")
    for (i, raw) in lines.enumerated() {
        var line = raw
        var heading = false
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let indent = String(repeating: " ", count: (line.count - line.drop(while: { $0 == " " }).count) / 2 * 3)
        if trimmed.hasPrefix("#") {
            line = String(trimmed.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)
            heading = true
        } else if trimmed.hasPrefix("* ") || trimmed.hasPrefix("- ") || trimmed.hasPrefix("• ") {
            line = indent + "•  " + trimmed.dropFirst(2).trimmingCharacters(in: .whitespaces)
        }
        var piece = (try? AttributedString(markdown: line, options: options)) ?? AttributedString(line)
        if heading { piece.font = .system(size: 13.5, weight: .semibold) }
        out.append(piece)
        if i < lines.count - 1 { out.append(AttributedString("\n")) }
    }
    return out
}

// MARK: - Panel UI

struct PanelView: View {
    @ObservedObject var model: ExplainModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if !model.readSummary.isEmpty {
                Text(model.readSummary).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(2)
                    .padding(.horizontal, 14).padding(.bottom, 8).frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider().opacity(0.5)
            content
            Divider().opacity(0.5)
            footer
        }
        .frame(minWidth: 380, idealWidth: 440, maxWidth: .infinity, minHeight: 260, maxHeight: .infinity)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "text.viewfinder").font(.system(size: 14, weight: .semibold)).foregroundStyle(.orange)
            Text("Mack").font(.system(size: 13, weight: .semibold))
            if model.busy { ProgressView().controlSize(.small).scaleEffect(0.7).frame(width: 14, height: 14) }
            Text(model.status).font(.system(size: 11.5)).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
            Button(action: model.onRecapture) { Image(systemName: "viewfinder") }
                .buttonStyle(.plain).foregroundStyle(.secondary).help("Select another area")
            Button(action: model.onClose) { Image(systemName: "xmark") }
                .buttonStyle(.plain).foregroundStyle(.secondary).help("Close (Esc)")
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var content: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if let notice = model.notice {
                        Text(notice).font(.system(size: 13)).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        Text(renderMarkdown(model.answer)).font(.system(size: 13.5)).lineSpacing(3)
                            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    }
                    if model.showCaptured && !model.captured.isEmpty {
                        Text(model.captured).font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
                            .textSelection(.enabled).padding(8).frame(maxWidth: .infinity, alignment: .leading)
                            .background(RoundedRectangle(cornerRadius: 6).fill(Color.primary.opacity(0.06)))
                    }
                    Color.clear.frame(height: 1).id("end")
                }
                .padding(14).frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: model.answer) { if model.busy { proxy.scrollTo("end", anchor: .bottom) } }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                chip("Simpler", .simpler)
                chip("More detail", .detail)
                chip("Summarize", .summary)
                Spacer()
                if !model.meta.isEmpty { Text(model.meta).font(.system(size: 10.5, design: .monospaced)).foregroundStyle(.tertiary) }
                Button(action: { model.showCaptured.toggle() }) { Image(systemName: "text.alignleft") }
                    .buttonStyle(.plain).foregroundStyle(model.showCaptured ? Color.orange : Color.secondary)
                    .help("Show the text Mack read from the screen").disabled(model.captured.isEmpty)
                Button(action: copyAnswer) { Image(systemName: "doc.on.doc") }
                    .buttonStyle(.plain).foregroundStyle(.secondary).help("Copy the explanation")
                    .disabled(model.answer.isEmpty)
            }
            TextField("Ask something about this selection…", text: $model.question)
                .textFieldStyle(.plain).font(.system(size: 12.5))
                .padding(.horizontal, 9).padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 7).fill(Color.primary.opacity(0.07)))
                .disabled(model.captured.isEmpty)
                .onSubmit {
                    let q = model.question.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !q.isEmpty { model.onAsk(q) }
                }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private func chip(_ title: String, _ style: Style) -> some View {
        Button(title) { model.onStyle(style) }
            .buttonStyle(.plain).font(.system(size: 11.5, weight: .medium))
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(Capsule().fill(Color.primary.opacity(0.08)))
            .disabled(model.captured.isEmpty)
    }

    private func copyAnswer() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(model.answer, forType: .string)
    }
}

final class FloatingPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

// MARK: - Server process and streaming

final class MackServer {
    private var process: Process?
    let root: URL?

    init() {
        // The project folder is recorded at build time; fall back to walking up from the bundle.
        if let path = Bundle.main.object(forInfoDictionaryKey: "MackRoot") as? String,
           FileManager.default.fileExists(atPath: path + "/mack.toml") {
            root = URL(fileURLWithPath: path)
        } else {
            var dir = Bundle.main.bundleURL
            var found: URL? = nil
            for _ in 0..<5 {
                dir.deleteLastPathComponent()
                if FileManager.default.fileExists(atPath: dir.appendingPathComponent("mack.toml").path) { found = dir; break }
            }
            root = found
        }
    }

    enum State { case ready, loading, down, outdated, failed(String) }

    func state() async -> State {
        var request = URLRequest(url: serverURL.appendingPathComponent("api/status"))
        request.timeoutInterval = 2
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              let http = response as? HTTPURLResponse else { return .down }
        if http.statusCode == 503 { return .loading }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return .down }
        if let error = json["error"] as? String { return .failed(error) }
        let features = json["features"] as? [String] ?? []
        return features.contains("explain") ? .ready : .outdated
    }

    /// Start the local engine if nothing is listening. Returns an error message on failure.
    func launch() -> String? {
        if let p = process, p.isRunning { return nil }
        guard let root = root else { return "Mack can't find its project folder (the one with mack.toml)." }
        let python = root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            return "Mack's Python environment is missing at \(python.path)."
        }
        let p = Process()
        p.executableURL = python
        p.arguments = ["-m", "mack.cli", "serve"]
        p.currentDirectoryURL = root
        let logURL = root.appendingPathComponent("state/server.log")
        try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        if let log = try? FileHandle(forWritingTo: logURL) {
            p.standardOutput = log
            p.standardError = log
        }
        do { try p.run() } catch { return "Could not start Mack's engine: \(error.localizedDescription)" }
        process = p
        return nil
    }

    func stop() {
        if let p = process, p.isRunning { p.terminate() }
    }
}

enum ExplainEvent {
    case token(String)
    case done(tps: Double, clipped: Bool)
    case failure(String)
}

/// POST the captured text to the engine and deliver its server-sent events in order.
func streamExplain(text: String, style: Style, question: String?, onEvent: @escaping (ExplainEvent) -> Void) async throws {
    var request = URLRequest(url: serverURL.appendingPathComponent("api/explain"))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    var body: [String: Any] = ["text": text, "style": style.rawValue]
    if let q = question { body["question"] = q }
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    request.timeoutInterval = 300
    let (bytes, _) = try await URLSession.shared.bytes(for: request)
    for try await line in bytes.lines {
        guard line.hasPrefix("data: "), let data = line.dropFirst(6).data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = event["type"] as? String else { continue }
        switch type {
        case "token": onEvent(.token(event["text"] as? String ?? ""))
        case "done": onEvent(.done(tps: event["tps"] as? Double ?? 0, clipped: event["clipped"] as? Bool ?? false))
        case "error": onEvent(.failure(event["message"] as? String ?? "Something went wrong."))
        default: break
        }
    }
}

// MARK: - Background tasks

struct TaskInfo {
    var id = "", title = "", status = "", stage = "", current = "", outDir = ""
    var progress = 0.0, done = 0, total = 0

    init(_ d: [String: Any]) {
        id = d["id"] as? String ?? ""
        title = d["title"] as? String ?? ""
        status = d["status"] as? String ?? ""
        stage = d["stage"] as? String ?? ""
        current = d["current"] as? String ?? ""
        outDir = d["out_dir"] as? String ?? ""
        progress = d["progress"] as? Double ?? 0
        done = d["units_done"] as? Int ?? 0
        total = d["units_total"] as? Int ?? 0
    }
}

func fetchTasks() async -> [TaskInfo]? {
    var request = URLRequest(url: serverURL.appendingPathComponent("api/tasks"))
    request.timeoutInterval = 2
    guard let (data, _) = try? await URLSession.shared.data(for: request),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let list = json["tasks"] as? [[String: Any]] else { return nil }
    return list.map(TaskInfo.init)
}

func postJSON(_ path: String, _ body: [String: Any]) async -> (Int, [String: Any]) {
    var request = URLRequest(url: serverURL.appendingPathComponent(path))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)
    request.timeoutInterval = 5
    guard let (data, response) = try? await URLSession.shared.data(for: request) else { return (0, [:]) }
    let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    return ((response as? HTTPURLResponse)?.statusCode ?? 0, json)
}

final class NewTaskModel: ObservableObject {
    @Published var prompt = ""
    @Published var kind = "auto"
    @Published var pages = 8
    @Published var pace = "auto"
    @Published var error = ""
    @Published var sending = false
    var onSubmit: () -> Void = {}
    var onCancel: () -> Void = {}
}

struct NewTaskView: View {
    @ObservedObject var model: NewTaskModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("New background task").font(.system(size: 15, weight: .semibold))
            Text("Describe something big. Mack plans it, then works through it slowly while you do other things.")
                .font(.system(size: 12)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            TextEditor(text: $model.prompt)
                .font(.system(size: 13)).frame(minHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3)))
            HStack {
                Picker("Kind", selection: $model.kind) {
                    Text("Decide for me").tag("auto"); Text("Report").tag("report"); Text("Codebase").tag("code")
                }.frame(maxWidth: 220)
                Stepper("Pages: \(model.pages)", value: $model.pages, in: 1...60).disabled(model.kind == "code")
            }
            Picker("Pace", selection: $model.pace) {
                Text("Auto").tag("auto"); Text("Gentle").tag("gentle"); Text("Balanced").tag("balanced"); Text("Steady").tag("steady"); Text("Full").tag("full")
            }.pickerStyle(.segmented)
            Text(paceHint).font(.system(size: 11.5)).foregroundStyle(.secondary)
            if !model.error.isEmpty { Text(model.error).font(.system(size: 12)).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel", action: model.onCancel).keyboardShortcut(.cancelAction)
                Button(model.sending ? "Starting…" : "Start Task", action: model.onSubmit)
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.sending || model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(18).frame(width: 480)
    }

    private var paceHint: String {
        switch model.pace {
        case "auto": return "Full speed while you're away from the Mac, gentle while your apps are busy, paused if it gets hot."
        case "gentle": return "Slowest. Keeps Mack's effect on other apps to about 5%."
        case "balanced": return "The default. About a 10% effect on memory-heavy apps, a few tokens per second."
        case "steady": return "About twice as fast as balanced, still capped."
        default: return "No cap. Fastest, and you will notice it."
        }
    }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private let model = ExplainModel()
    private let server = MackServer()
    private var panel: FloatingPanel?
    private var stream: Task<Void, Never>?
    private var capturing = false
    private var hotKey: EventHotKeyRef?
    private var tasks: [TaskInfo] = []
    private var finishedUnseen = false
    private var newTaskWindow: NSWindow?
    private let newTask = NewTaskModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "checklist", accessibilityDescription: "Mack")
            button.imagePosition = .imageLeading
            button.toolTip = "Mack: background tasks"
            button.target = self
            button.action = #selector(statusClicked)
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
        model.onStyle = { [weak self] style in self?.ask(style: style, question: nil) }
        model.onAsk = { [weak self] q in self?.ask(style: .explain, question: q) }
        model.onClose = { [weak self] in self?.closePanel() }
        model.onRecapture = { [weak self] in self?.startCapture() }
        newTask.onSubmit = { [weak self] in self?.submitTask() }
        newTask.onCancel = { [weak self] in self?.newTaskWindow?.orderOut(nil) }
        registerHotKey()
        // The engine runs as long as the app does: background tasks need it.
        Task { if case .down = await server.state() { _ = server.launch() } }
        Timer.scheduledTimer(withTimeInterval: 8, repeats: true) { [weak self] _ in self?.pollTasks() }
        pollTasks()
    }

    private func pollTasks() {
        Task { @MainActor in
            guard let list = await fetchTasks() else { return }
            let wasRunning = Set(tasks.filter { $0.status == "running" }.map(\.id))
            tasks = list
            if list.contains(where: { wasRunning.contains($0.id) && $0.status == "done" }) { finishedUnseen = true }
            updateStatusTitle()
        }
    }

    private func updateStatusTitle() {
        guard let button = statusItem.button else { return }
        if let running = tasks.first(where: { $0.status == "running" }) {
            button.title = " \(Int(running.progress * 100))%"
        } else if finishedUnseen {
            button.title = " ✓"
        } else {
            button.title = ""
        }
    }

    @objc private func showNewTask() {
        if newTaskWindow == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 480, height: 360), styleMask: [.titled, .closable],
                             backing: .buffered, defer: false)
            w.title = "Mack"
            w.isReleasedWhenClosed = false
            w.contentView = NSHostingView(rootView: NewTaskView(model: newTask))
            w.center()
            newTaskWindow = w
        }
        newTask.error = ""
        NSApp.activate(ignoringOtherApps: true)
        newTaskWindow?.makeKeyAndOrderFront(nil)
    }

    private func submitTask() {
        newTask.sending = true
        newTask.error = ""
        let body: [String: Any] = ["prompt": newTask.prompt, "kind": newTask.kind, "pages": newTask.pages, "pace": newTask.pace]
        Task { @MainActor in
            if case .down = await server.state() { _ = server.launch() }
            var (status, json) = await postJSON("api/tasks", body)
            var tries = 0
            while status == 0 && tries < 20 {  // engine still starting
                try? await Task.sleep(nanoseconds: 500_000_000)
                (status, json) = await postJSON("api/tasks", body)
                tries += 1
            }
            newTask.sending = false
            if status == 200 {
                newTask.prompt = ""
                newTaskWindow?.orderOut(nil)
                pollTasks()
                openDashboard()
            } else {
                newTask.error = (json["error"] as? String) ?? "Mack's engine is not reachable."
            }
        }
    }

    @objc private func openDashboard() {
        NSWorkspace.shared.open(serverURL)
    }

    @objc private func controlTask(_ sender: NSMenuItem) {
        guard let info = sender.representedObject as? [String] else { return }
        Task { @MainActor in
            _ = await postJSON("api/tasks/\(info[0])/\(info[1])", [:])
            pollTasks()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        server.stop()
    }

    // MARK: status item

    @objc private func statusClicked() {
        finishedUnseen = false
        updateStatusTitle()
        let menu = NSMenu()
        let active = tasks.filter { ["running", "queued", "paused"].contains($0.status) }
        if active.isEmpty {
            let none = menu.addItem(withTitle: tasks.isEmpty ? "No tasks yet" : "No tasks running", action: nil, keyEquivalent: "")
            none.isEnabled = false
        }
        for t in active.prefix(5) {
            let head = menu.addItem(withTitle: "\(t.title.prefix(48))  ·  \(Int(t.progress * 100))%", action: nil, keyEquivalent: "")
            head.isEnabled = false
            let detail = t.status == "running"
                ? "\(t.stage.capitalized) \(t.total > 0 ? "\(min(t.done + 1, t.total))/\(t.total)" : ""): \(t.current.prefix(40))"
                : t.status.capitalized
            let line = menu.addItem(withTitle: "    " + detail, action: nil, keyEquivalent: "")
            line.isEnabled = false
            let action = t.status == "paused" ? "resume" : "pause"
            let toggle = menu.addItem(withTitle: "    \(action.capitalized)", action: #selector(controlTask(_:)), keyEquivalent: "")
            toggle.target = self
            toggle.representedObject = [t.id, action]
        }
        if let last = tasks.first(where: { $0.status == "done" }) {
            menu.addItem(.separator())
            let item = menu.addItem(withTitle: "Open Last Result: \(last.title.prefix(36))", action: #selector(controlTask(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = [last.id, "open"]
        }
        menu.addItem(.separator())
        menu.addItem(withTitle: "New Task…", action: #selector(showNewTask), keyEquivalent: "n").target = self
        menu.addItem(withTitle: "Open Dashboard", action: #selector(openDashboard), keyEquivalent: "d").target = self
        menu.addItem(.separator())
        let explain = NSMenu()
        explain.addItem(withTitle: "Explain an Area of the Screen  (⌃⌥M)", action: #selector(startCapture), keyEquivalent: "").target = self
        explain.addItem(withTitle: "Explain Copied Text", action: #selector(explainClipboard), keyEquivalent: "").target = self
        explain.addItem(withTitle: "Check Screen Access", action: #selector(checkScreenAccess), keyEquivalent: "").target = self
        let explainItem = menu.addItem(withTitle: "Explain", action: nil, keyEquivalent: "")
        menu.setSubmenu(explain, for: explainItem)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Mack (tasks resume next time)", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
        pollTasks()
    }

    private func registerHotKey() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, userData in
            let me = Unmanaged<AppDelegate>.fromOpaque(userData!).takeUnretainedValue()
            DispatchQueue.main.async { me.startCapture() }
            return noErr
        }, 1, &spec, Unmanaged.passUnretained(self).toOpaque(), nil)
        let id = EventHotKeyID(signature: OSType(0x4D41_434B), id: 1)  // 'MACK'
        RegisterEventHotKey(UInt32(kVK_ANSI_M), UInt32(controlKey | optionKey), id, GetApplicationEventTarget(), 0, &hotKey)
    }

    // MARK: capture

    @objc func startCapture() {
        guard !capturing else { return }
        if !CGPreflightScreenCaptureAccess() || !canSeeOtherWindows() {
            CGRequestScreenCaptureAccess()  // macOS shows its own prompt the first time
            showPanel()
            present(notice: "Mack can't read the screen yet: macOS is not letting it see other apps' windows.\n\n" + permissionSteps,
                    status: "Permission needed")
            return
        }
        capturing = true
        stream?.cancel()
        panel?.orderOut(nil)  // keep the old panel out of the screenshot
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("mack-\(UUID().uuidString).png")
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        task.arguments = ["-i", "-s", "-x", "-t", "png", file.path]  // system crosshair; Esc cancels
        task.terminationHandler = { [weak self] _ in DispatchQueue.main.async { self?.finishCapture(file) } }
        do { try task.run() } catch { capturing = false }
    }

    private func finishCapture(_ file: URL) {
        capturing = false
        defer { try? FileManager.default.removeItem(at: file) }  // the screenshot never outlives the OCR
        guard let image = loadCGImage(file) else { return }  // Esc: nothing was captured
        showPanel()
        model.readSummary = ""
        present(status: "Reading the screen…", busy: true)
        let size = "\(image.width)×\(image.height)"
        if debugEnabled { debugSave(image: file, note: "captured \(size) px\n") }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let text = recognizeText(in: image)
            DispatchQueue.main.async {
                guard let self = self else { return }
                if self.debugEnabled { self.debugAppend("--- text read (\(text.count) characters) ---\n\(text)\n") }
                self.gotText(text, areaSize: size, emptyNotice: """
                    I captured a \(size) pixel area but found no text in it.

                    If there was text there, macOS probably gave Mack only the desktop picture instead of the windows. Right-click the Mack icon and choose “Check Screen Access” to find out.
                    """)
            }
        }
    }

    // MARK: debug record (off unless `defaults write ai.mack.explainer Debug -bool YES`)

    private var debugEnabled: Bool { UserDefaults.standard.bool(forKey: "Debug") }
    private var debugDir: URL? { server.root?.appendingPathComponent("state/debug") }

    /// Keeps the last capture and what was read from it, locally, so a wrong result can be examined.
    private func debugSave(image: URL, note: String) {
        guard let dir = debugDir else { return }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let copy = dir.appendingPathComponent("last_capture.png")
        try? FileManager.default.removeItem(at: copy)
        try? FileManager.default.copyItem(at: image, to: copy)
        try? note.write(to: dir.appendingPathComponent("last_explain.txt"), atomically: true, encoding: .utf8)
    }

    private func debugAppend(_ text: String) {
        guard let url = debugDir?.appendingPathComponent("last_explain.txt"),
              let handle = try? FileHandle(forWritingTo: url) else { return }
        handle.seekToEndOfFile()
        handle.write(text.data(using: .utf8) ?? Data())
        try? handle.close()
    }

    @objc private func checkScreenAccess() {
        stream?.cancel()
        model.captured = ""
        model.readSummary = ""
        panel?.orderOut(nil)  // the test capture should see the screen, not this panel
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let d = ScreenDiagnosis.run()
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.showPanel()
                let verdict = d.works
                    ? "Screen access is working. Mack can read text from your screen."
                    : "Screen access is NOT working: Mack is not being given what is on screen.\n\n" + permissionSteps
                self.present(notice: verdict + "\n\n" + d.report, status: d.works ? "Screen access OK" : "Permission needed")
            }
        }
    }

    @objc private func explainClipboard() {
        let text = NSPasteboard.general.string(forType: .string) ?? ""
        showPanel()
        gotText(text, emptyNotice: "There is no text on the clipboard. Copy some text first.")
    }

    private func gotText(_ text: String, areaSize: String? = nil,
                         emptyNotice: String = "I couldn't find any text in that area. Try dragging a slightly larger box around it.") {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        model.captured = cleaned
        model.showCaptured = false
        model.question = ""
        let words = cleaned.split(whereSeparator: { $0.isWhitespace }).count
        let preview = cleaned.replacingOccurrences(of: "\n", with: " ").prefix(90)
        model.readSummary = cleaned.isEmpty ? "" :
            "Read \(words) word\(words == 1 ? "" : "s")\(areaSize.map { " from a \($0) px area" } ?? ""): “\(preview)\(cleaned.count > 90 ? "…" : "")”"
        if cleaned.count < 2 {
            present(notice: emptyNotice, status: "")
            return
        }
        ask(style: .explain, question: nil)
    }

    // MARK: asking the engine

    private func present(notice: String? = nil, status: String, busy: Bool = false) {
        model.notice = notice
        model.status = status
        model.busy = busy
        if notice != nil || busy { model.answer = ""; model.meta = "" }
    }

    private func ask(style: Style, question: String?) {
        guard !model.captured.isEmpty else { return }
        stream?.cancel()
        present(status: "Thinking…", busy: true)
        let text = model.captured
        stream = Task { [weak self] in await self?.run(text: text, style: style, question: question) }
    }

    @MainActor
    private func run(text: String, style: Style, question: String?) async {
        // Make sure the engine is up; start it if this app is the first thing to need it.
        var waited = 0
        engine: while true {
            switch await server.state() {
            case .ready: break engine
            case .outdated:
                present(notice: "An older Mack server is running and doesn't know how to explain selections. Stop it (Ctrl-C in its terminal), then try again and Mack will start the current one.", status: "Restart needed")
                return
            case .failed(let message):
                present(notice: "Mack's engine failed to start: \(message)", status: "Error")
                return
            case .down:
                if let error = server.launch() { present(notice: error, status: "Error"); return }
                model.status = "Starting Mack's model…"
            case .loading:
                model.status = "Loading the model…"
            }
            waited += 1
            if waited > 120 { present(notice: "Mack's engine did not come up. See state/server.log in the Mack folder.", status: "Error"); return }
            try? await Task.sleep(nanoseconds: 500_000_000)
            if Task.isCancelled { return }
        }

        model.status = question == nil ? "Explaining…" : "Answering…"
        var first = true
        var failed = false
        do {
            try await streamExplain(text: text, style: style, question: question) { [weak self] event in
                guard let self = self else { return }
                switch event {
                case .token(let piece):
                    if first { self.model.answer = ""; first = false }
                    self.model.answer += piece
                case .done(let tps, let clipped):
                    self.model.meta = String(format: "%.0f tok/s", tps) + (clipped ? " · long selection, middle skipped" : "")
                case .failure(let message):
                    failed = true
                    self.present(notice: message, status: "")
                }
            }
            if !failed {
                model.status = ""
                model.busy = false
                if debugEnabled { debugAppend("--- Mack's reply ---\n\(model.answer)\n") }
            }
        } catch {
            if Task.isCancelled { return }  // replaced by a newer request, or the panel was closed
            present(notice: "Lost the connection to Mack's engine: \(error.localizedDescription)", status: "Error")
        }
    }

    // MARK: panel

    private func showPanel() {
        if panel == nil {
            let p = FloatingPanel(contentRect: NSRect(x: 0, y: 0, width: 440, height: 420),
                                  styleMask: [.titled, .closable, .resizable, .fullSizeContentView, .nonactivatingPanel],
                                  backing: .buffered, defer: false)
            p.titleVisibility = .hidden
            p.titlebarAppearsTransparent = true
            p.isMovableByWindowBackground = true
            p.isFloatingPanel = true
            p.level = .floating
            p.hidesOnDeactivate = false
            p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            p.standardWindowButton(.closeButton)?.isHidden = true
            p.standardWindowButton(.miniaturizeButton)?.isHidden = true
            p.standardWindowButton(.zoomButton)?.isHidden = true
            p.isReleasedWhenClosed = false
            let blur = NSVisualEffectView()
            blur.material = .popover
            blur.state = .active
            blur.blendingMode = .behindWindow
            let host = NSHostingView(rootView: PanelView(model: model))
            host.translatesAutoresizingMaskIntoConstraints = false
            blur.addSubview(host)
            NSLayoutConstraint.activate([
                host.topAnchor.constraint(equalTo: blur.topAnchor), host.bottomAnchor.constraint(equalTo: blur.bottomAnchor),
                host.leadingAnchor.constraint(equalTo: blur.leadingAnchor), host.trailingAnchor.constraint(equalTo: blur.trailingAnchor),
            ])
            p.contentView = blur
            panel = p
        }
        guard let panel = panel else { return }
        if !panel.isVisible {
            // Next to where the drag ended, kept on screen.
            let mouse = NSEvent.mouseLocation
            let screen = NSScreen.screens.first { $0.frame.contains(mouse) } ?? NSScreen.main
            let visible = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            let size = panel.frame.size
            var origin = NSPoint(x: mouse.x + 16, y: mouse.y - size.height - 16)
            if origin.x + size.width > visible.maxX { origin.x = mouse.x - size.width - 16 }
            origin.x = min(max(origin.x, visible.minX + 8), visible.maxX - size.width - 8)
            origin.y = min(max(origin.y, visible.minY + 8), visible.maxY - size.height - 8)
            panel.setFrameOrigin(origin)
        }
        panel.orderFrontRegardless()
    }

    private func closePanel() {
        stream?.cancel()  // dropping the connection stops generation on the engine
        model.busy = false
        panel?.orderOut(nil)
    }
}

// MARK: - Entry point and test modes

func renderTextImage(_ lines: [String]) -> CGImage? {
    let size = NSSize(width: 900, height: 60 + 44 * lines.count)
    guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
                                     bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                     colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { return nil }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSColor.white.setFill()
    NSRect(origin: .zero, size: size).fill()
    let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 26), .foregroundColor: NSColor.black]
    for (i, line) in lines.enumerated() {
        (line as NSString).draw(at: NSPoint(x: 30, y: size.height - 60 - CGFloat(i) * 44), withAttributes: attrs)
    }
    NSGraphicsContext.restoreGraphicsState()
    return rep.cgImage
}

/// Text drawn the way a tight drag captures it: the canvas hugs the glyphs.
func renderTight(_ lines: [String], fontSize: CGFloat, margin: CGFloat, dark: Bool) -> CGImage? {
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: fontSize), .foregroundColor: dark ? NSColor(white: 0.92, alpha: 1) : NSColor.black,
    ]
    let lineHeight = ceil(fontSize * 1.35)
    let width = lines.map { ($0 as NSString).size(withAttributes: attrs).width }.max() ?? 10
    let size = NSSize(width: ceil(width + 2 * margin), height: lineHeight * CGFloat(lines.count) + 2 * margin)
    guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
                                     bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                     colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { return nil }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    (dark ? NSColor(white: 0.12, alpha: 1) : NSColor.white).setFill()
    NSRect(origin: .zero, size: size).fill()
    for (i, line) in lines.enumerated() {
        (line as NSString).draw(at: NSPoint(x: margin, y: size.height - margin - lineHeight * CGFloat(i + 1) + fontSize * 0.2), withAttributes: attrs)
    }
    NSGraphicsContext.restoreGraphicsState()
    return rep.cgImage
}

/// Text of a PDF: its own text layer when it has one, otherwise OCR of the first pages.
func pdfText(_ url: URL, maxOCRPages: Int = 3) -> String? {
    guard let doc = PDFDocument(url: url) else { return nil }
    if let layer = doc.string, layer.trimmingCharacters(in: .whitespacesAndNewlines).count >= 20 {
        return layer
    }
    var pages: [String] = []
    for i in 0..<min(doc.pageCount, maxOCRPages) {
        guard let page = doc.page(at: i) else { continue }
        let bounds = page.bounds(for: .mediaBox)
        let image = page.thumbnail(of: NSSize(width: bounds.width * 2, height: bounds.height * 2), for: .mediaBox)
        if let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            pages.append(recognizeText(in: cg))
        }
    }
    return pages.joined(separator: "\n\n")
}

func writePNG(_ image: CGImage, to path: String) -> Bool {
    let rep = NSBitmapImageRep(cgImage: image)
    guard let data = rep.representation(using: .png, properties: [:]) else { return false }
    return (try? data.write(to: URL(fileURLWithPath: path))) != nil
}

func makePDF(_ path: String, text: String, scan: Bool) -> Bool {
    var box = CGRect(x: 0, y: 0, width: 612, height: 792)
    guard let ctx = CGContext(URL(fileURLWithPath: path) as CFURL, mediaBox: &box, nil) else { return false }
    ctx.beginPDFPage(nil)
    if scan, let image = renderTextImage(text.components(separatedBy: "\n")) {
        let w = min(560.0, CGFloat(image.width) / 2), h = CGFloat(image.height) * w / CGFloat(image.width)
        ctx.draw(image, in: CGRect(x: 26, y: 792 - 40 - h, width: w, height: h))
    } else {
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
        let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 13), .foregroundColor: NSColor.black]
        (text as NSString).draw(in: CGRect(x: 54, y: 54, width: 504, height: 684), withAttributes: attrs)
        NSGraphicsContext.restoreGraphicsState()
    }
    ctx.endPDFPage()
    ctx.closePDF()
    return true
}

let arguments = CommandLine.arguments
if let i = arguments.firstIndex(of: "--pdf-text"), i + 1 < arguments.count {
    guard let text = pdfText(URL(fileURLWithPath: arguments[i + 1])) else { print("could not read PDF"); exit(1) }
    print(text)
    exit(0)
} else if let i = arguments.firstIndex(of: "--make-image"), i + 2 < arguments.count {
    guard let image = renderTextImage(arguments[i + 2].components(separatedBy: "\n")), writePNG(image, to: arguments[i + 1]) else { exit(1) }
    exit(0)
} else if let i = arguments.firstIndex(of: "--make-pdf"), i + 2 < arguments.count {
    exit(makePDF(arguments[i + 1], text: arguments[i + 2], scan: arguments.contains("--scan")) ? 0 : 1)
} else if let i = arguments.firstIndex(of: "--diagnose"), i + 1 < arguments.count {
    let d = ScreenDiagnosis.run()
    try? ("works: \(d.works ? "yes" : "no")\n" + d.report + "\n").write(toFile: arguments[i + 1], atomically: true, encoding: .utf8)
    exit(0)
} else if arguments.contains("--ocr-matrix") {
    let lines = ["The Krebs cycle oxidizes acetyl-CoA to CO2."]
    for dark in [false, true] {
        for fontSize in [11.0, 13.0, 16.0, 26.0] as [CGFloat] {
            var row = "\(dark ? "dark " : "light") \(Int(fontSize))pt:"
            for margin in [0.0, 2.0, 8.0, 30.0] as [CGFloat] {
                guard let image = renderTight(lines, fontSize: fontSize, margin: margin, dark: dark) else { continue }
                let text = recognizeText(in: image)
                row += "  m\(Int(margin))=\(text.contains("Krebs") ? "ok" : (text.isEmpty ? "EMPTY" : "wrong"))(\(image.width)x\(image.height))"
            }
            print(row)
        }
    }
    exit(0)
} else if arguments.contains("--selftest") {
    let expected = ["The Krebs cycle oxidizes acetyl-CoA to CO2.", "Late payment fee up to $40.", "const port = process.env.PORT ?? 3000;"]
    guard let image = renderTextImage(expected) else { print("FAIL could not render"); exit(1) }
    let text = recognizeText(in: image)
    print(text)
    let ok = text.contains("Krebs cycle") && text.contains("$40") && text.contains("3000")
        && text.components(separatedBy: "\n").count == expected.count
    print(ok ? "PASS ocr" : "FAIL ocr")
    exit(ok ? 0 : 1)
} else if let i = arguments.firstIndex(of: "--ocr"), i + 1 < arguments.count {
    guard let image = loadCGImage(URL(fileURLWithPath: arguments[i + 1])) else { print("could not read image"); exit(1) }
    print(recognizeText(in: image))
    exit(0)
} else if let i = arguments.firstIndex(of: "--explain-text"), i + 1 < arguments.count {
    let text = arguments[i + 1]
    Task {
        switch await MackServer().state() {
        case .ready: break
        case let other: print("engine not ready: \(other)"); exit(2)
        }
        var pieces = 0
        do {
            try await streamExplain(text: text, style: .explain, question: nil) { event in
                switch event {
                case .token(let piece): pieces += 1; print(piece, terminator: ""); fflush(stdout)
                case .done(let tps, _): print("\n[done: \(pieces) streamed pieces, \(tps) tok/s]")
                case .failure(let message): print("\n[error: \(message)]")
                }
            }
            exit(pieces > 0 ? 0 : 1)
        } catch { print("request failed: \(error)"); exit(1) }
    }
    RunLoop.main.run()
} else if let i = arguments.firstIndex(of: "--render-newtask"), i + 1 < arguments.count {
    let app = NSApplication.shared
    app.setActivationPolicy(.prohibited)
    let m = NewTaskModel()
    m.prompt = "Write a 15-page report on the history and future of nuclear fusion, for a general audience."
    let host = NSHostingView(rootView: NewTaskView(model: m).background(Color(nsColor: .windowBackgroundColor)))
    host.frame = NSRect(x: 0, y: 0, width: 480, height: 380)
    let window = NSWindow(contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.contentView = host
    RunLoop.main.run(until: Date().addingTimeInterval(0.5))
    guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { exit(1) }
    host.cacheDisplay(in: host.bounds, to: rep)
    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: arguments[i + 1]))
    print("wrote \(arguments[i + 1])")
    exit(0)
} else if let i = arguments.firstIndex(of: "--render-test"), i + 1 < arguments.count {
    let app = NSApplication.shared
    app.setActivationPolicy(.prohibited)
    let model = ExplainModel()
    model.captured = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    model.answer = "This error means the program ran out of **video memory** while computing the loss.\n\n- **Cause**: the batch is too large for an 8 GB GPU.\n- **Fix**: lower `batch_size`, or use a smaller model.\n\n## Check\nMake sure no other training run is using the GPU."
    model.meta = "35 tok/s"
    model.readSummary = "Read 11 words from a 812×96 px area: “RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB”"
    model.showCaptured = true
    let host = NSHostingView(rootView: PanelView(model: model).background(Color(nsColor: .windowBackgroundColor)))
    host.frame = NSRect(x: 0, y: 0, width: 440, height: 420)
    let window = NSWindow(contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.contentView = host
    host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.5))
    guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { print("FAIL no bitmap"); exit(1) }
    host.cacheDisplay(in: host.bounds, to: rep)
    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: arguments[i + 1]))
    print("wrote \(arguments[i + 1])")
    exit(0)
} else {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.accessory)  // menu bar only, no Dock icon
    app.run()
}
