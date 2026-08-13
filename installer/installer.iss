; ScreenMind installer — embedded Python runtime, no system Python required
; Build: ISCC.exe installer.iss   (expects installer/runtime/ assembled — see
; .github/workflows/installer.yml, which builds it from scratch)

#define MyAppName "ScreenMind"
; Overridable from the CLI: ISCC /DMyAppVersion=x.y.z (see installer.yml)
#ifndef MyAppVersion
#define MyAppVersion "0.2.0+local"
#endif
#define MyAppPublisher "valentinvvv"
#define MyAppURL "https://github.com/valentinvvv/ScreenMind"
#ifndef RuntimeDir
#define RuntimeDir "runtime"
#endif

[Setup]
AppId={{8F4C2E61-9B3A-4D5E-A7F0-1C2D3E4F5A60}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=ScreenMind-{#MyAppVersion}-setup-win64
SetupIconFile=..\screenmind\assets\favicon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\pythonw.exe
UninstallDisplayName={#MyAppName} {#MyAppVersion}
DirExistsWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Entire embedded Python runtime with screenmind + core deps pre-installed
Source: "{#RuntimeDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\pythonw.exe"; Parameters: "-m screenmind.launcher"; WorkingDir: "{app}"; IconFilename: "{app}\Lib\site-packages\screenmind\assets\favicon.ico"; Comment: "Start ScreenMind and open the dashboard"
Name: "{group}\{#MyAppName} Console"; Filename: "{app}\python.exe"; Parameters: "-m screenmind"; WorkingDir: "{app}"; IconFilename: "{app}\Lib\site-packages\screenmind\assets\favicon.ico"; Comment: "Start ScreenMind with a console window (debug)"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\pythonw.exe"; Parameters: "-m screenmind.launcher"; WorkingDir: "{app}"; IconFilename: "{app}\Lib\site-packages\screenmind\assets\favicon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\pythonw.exe"; Parameters: "-m screenmind.launcher"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove bytecode caches so uninstall leaves nothing behind in {app}
Type: filesandordirs; Name: "{app}\Lib\site-packages\*\__pycache__"
