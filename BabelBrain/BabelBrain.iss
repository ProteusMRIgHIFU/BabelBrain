; BabelBrain — Windows installer (Inno Setup)
; Build:  ISCC.exe /DAppVersion=<version> /DBuildId=<version+commit> BabelBrain.iss
;
; Two-app model. Installs:
;   {app}\BabelBrain.exe                                  - the launcher / main app
;   {app}\VersionSelector\BabelBrain-Version-Selector.exe - the version picker
; and seeds a default BabelBrain version into the per-user store:
;   {localappdata}\BabelBrain\versions\<BuildId>\BabelBrain.exe
; recording it as the default version for the Version Selector in:
;   {localappdata}\BabelBrain\default_build.json
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

[Code]
// Record the build this installer just seeded so the Version Selector adopts it
// as the default. Without it the new version installs but the Hub keeps running
// whatever was selected before. Mirrors the macOS PKG postinstall written by
// Hub/make_pkg_scripts.sh; read by Hub/state.py:adopt_installer_default.
procedure CurStepChanged(CurStep: TSetupStep);
var
  MarkerDir, Marker, Content: String;
begin
  if CurStep = ssPostInstall then
  begin
    MarkerDir := ExpandConstant('{localappdata}\BabelBrain');
    if ForceDirectories(MarkerDir) then
    begin
      Marker := MarkerDir + '\default_build.json';
      Content := '{' + #13#10 +
                 '  "build_id": "{#BuildId}",' + #13#10 +
                 '  "installed_at": "' +
                     GetDateTimeString('yyyy-mm-dd hh:nn:ss', '-', ':') +
                     '",' + #13#10 +
                 '  "source": "inno"' + #13#10 +
                 '}' + #13#10;
      // A failed marker write is not worth failing the install over.
      SaveStringToFile(Marker, Content, False);
    end;
  end;
end;
