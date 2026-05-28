; BabelBrain — Windows installer (Inno Setup)
; Build:  ISCC.exe /DAppVersion=<version> BabelBrain.iss
; Expects PyInstaller's onedir output at .\dist\BabelBrain\

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName       "BabelBrain"
#define AppPublisher  "Samuel Pichardo"
#define AppURL        "https://github.com/ProteusMRIgHIFU/BabelBrain"
#define AppExeName    "BabelBrain.exe"

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
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Pull the entire onedir bundle. recursesubdirs/createallsubdirs replicates
; the directory tree under {app} (Program Files\BabelBrain by default).
Source: "dist\BabelBrain\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
