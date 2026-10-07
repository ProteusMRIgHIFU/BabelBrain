; BabelBrain — Windows installer (Inno Setup)
; Build:  ISCC.exe /DAppVersion=<version> /DBuildId=<version+commit> BabelBrain.iss
;
; Two-app model (Windows needs no uninstaller app of its own — the Inno
; uninstaller does that job). Installs:
;   {app}\BabelBrain.exe                                  - the launcher / main app
;   {app}\VersionSelector\BabelBrain-Version-Selector.exe - the version picker
; and seeds a default BabelBrain version into the per-user store:
;   {localappdata}\BabelBrain\versions\<BuildId>\BabelBrain.exe
; recording it as the default version for the Version Selector in:
;   {localappdata}\BabelBrain\default_build.json
; Expects PyInstaller onedir output at .\dist\launcher\, .\dist\selector\,
; .\dist\version\ .
;
; Uninstall removes the seeded/downloaded versions too. They live in
; {localappdata}\BabelBrain, deliberately OUTSIDE {app}, so the stock
; uninstaller would leave several GB behind; CurUninstallStepChanged below
; hands that job to the Version Selector (--purge-user-data), which is the same
; code the macOS uninstaller app uses. Settings and custom transducers are
; small and often worth keeping, so they are removed only if the user says so.

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

[UninstallDelete]
; Backstop: if the Version Selector could not be run during uninstall (a
; partial install, a corrupt exe), the version store still goes. Inno processes
; these after the main file removal.
Type: filesandordirs; Name: "{localappdata}\BabelBrain"

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

// --------------------------------------------------------------------------
// Uninstall: leave nothing behind.
//
// {app} is Inno's to remove, but the version store is not — it lives in
// {localappdata}\BabelBrain so that versions survive an app upgrade. Running
// the Version Selector with --purge-user-data reuses Hub/uninstall.py, so
// Windows and macOS remove exactly the same set of paths. It runs at
// usUninstall, while the exe still exists, with no elevation of its own (a
// UAC prompt hidden behind the uninstall progress window would just hang), so
// a machine-wide %ProgramData% store is reported rather than silently skipped.
// --------------------------------------------------------------------------
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Selector, Params, Home: String;
  ResultCode: Integer;
  RemoveSettings, Purged: Boolean;
begin
  if CurUninstallStep <> usUninstall then
    Exit;

  RemoveSettings := MsgBox(
      'Also remove your BabelBrain settings and custom transducers?' + #13#10 + #13#10 +
      'These are small, and keeping them means a future reinstall finds your '
      + 'preferences and any transducers you created.' + #13#10 + #13#10 +
      'Your study data (images, .ini files and results) is never removed.',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;

  Purged := False;
  Selector := ExpandConstant('{app}\{#SelectorExe}');
  if FileExists(Selector) then
  begin
    Params := '--purge-user-data';
    if RemoveSettings then
      Params := Params + ' --include-settings';
    // SW_HIDE: no console flash. A non-zero exit means something could not be
    // removed (typically an all-users store needing administrator rights).
    if Exec(Selector, Params, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      Purged := ResultCode = 0;
      if ResultCode <> 0 then
        MsgBox('Some BabelBrain files could not be removed - usually versions '
             + 'installed for all users, which need administrator rights.'
             + #13#10 + #13#10 + 'You can delete them manually from:' + #13#10
             + ExpandConstant('{commonappdata}\BabelBrain'),
               mbInformation, MB_OK);
    end;
  end;

  // Fallback when the Version Selector is missing or did not start: remove the
  // same paths directly. [UninstallDelete] covers the version store as well.
  if not Purged then
  begin
    DelTree(ExpandConstant('{localappdata}\BabelBrain'), True, True, True);
    if RemoveSettings then
    begin
      Home := GetEnv('USERPROFILE');
      if Home <> '' then
      begin
        DelTree(Home + '\.config\BabelBrain', True, True, True);
        DelTree(Home + '\.babelbrain', True, True, True);
        DelTree(Home + '\.BabelBrainSync', True, True, True);
      end;
    end;
  end;
end;
