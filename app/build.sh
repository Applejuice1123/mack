#!/bin/bash
# Builds app/Mack.app, the menu-bar explainer. Needs only the Xcode command line tools.
#   app/build.sh          build
#   app/build.sh run      build, then open it
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
APP="Mack.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -swift-version 5 -target arm64-apple-macos14.0 Mack.swift -o "$APP/Contents/MacOS/Mack"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Mack</string>
    <key>CFBundleDisplayName</key><string>Mack</string>
    <key>CFBundleIdentifier</key><string>ai.mack.explainer</string>
    <key>CFBundleExecutable</key><string>Mack</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.2</string>
    <key>CFBundleVersion</key><string>2</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
    <key>MackRoot</key><string>$ROOT</string>
</dict>
</plist>
EOF

# Ad-hoc signature whose designated requirement is just the identifier. By default
# an ad-hoc build is identified by its exact hash, so every rebuild would silently
# lose the Screen Recording permission while System Settings still shows it on.
# The trade-off: any other ad-hoc app claiming this identifier would share the grant.
codesign --force --sign - --identifier ai.mack.explainer \
    --requirements '=designated => identifier "ai.mack.explainer"' "$APP" >/dev/null 2>&1

echo "built $ROOT/app/$APP"
if [ "${1:-}" = "run" ]; then open "$APP"; fi
