; BabelBrain — Windows installer (Inno Setup)
; Build:  ISCC.exe /DAppVersion=<version> /DBuildId=<version+commit> BabelBrain.iss
;
; Two-app model. Installs:
;   {app}\BabelBrain.exe                                  - the launcher / main app
;   {app}\VersionSelector\BabelBrain-Version-Selector.exe - the version picker
; and seeds a default BabelBrain version into the per-user store:
;   {localappdata}\BabelBrain\versions\<BuildId>\BabelBrain.exe
; Expects PyInstaller onedir output at .\dist\launcher\, .\dist\selector\,
; .\dist\version\ .

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef BuildId
  #define BuildId "0.0.0"
#endif

#define AppName       "BabelBrain"
#define AppPublisher  "Samuel Pichardo"
#define AppURL        "https://github.com/ProteusMRIgHIFU/BabelBrain"
#define AppExeName    "BabelBrain.exe"
#define SelectorName  "BabelBrain Version Selector"
#define SelectorExe   "VersionSelector\BabelBrain-Version-Selector.exe"

[Setup]
; Reusing the GUID from the previous WiX upgrade_guid keeps the brand consistent.
; (MSI UpgradeCode and Inno AppId are tracked independently, so this does not
; cross-upgrade old MSI installs — users on MSI need to uninstall it first.)
AppId={{b99cee55-c040-464d-8128-ae160c3bbd5e}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
; With PrivilegesRequired=lowest, {autopf} resolves to the per-user
; %LOCALAPPDATA%\Programs\BabelBrain so no admin rights are needed.
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE.rtf
OutputDir=.
OutputBaseFilename=BabelBrain-Setup
SetupIconFile=Proteus-Alciato-logo.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Install per-user by default (no elevation) so locked-down research machines
; can install without administrator rights. The user may still opt into a
; system-wide install via the elevation dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Launcher app (BabelBrain.exe) directly under {app}.
Source: "dist\launcher\BabelBrain\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Version Selector app in its own subfolder (separate onedir, avoids file clashes).
Source: "dist\selector\BabelBrain-Version-Selector\*"; DestDir: "{app}\VersionSelector"; Flags: ignoreversion recursesubdirs createallsubdirs
; Seed a default BabelBrain version into the per-user store so the app works
; offline immediately; the Version Selector can add/switch more later.
Source: "dist\version\BabelBrain\*"; DestDir: "{localappdata}\BabelBrain\versions\{#BuildId}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{#SelectorName}"; Filename: "{app}\{#SelectorExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
