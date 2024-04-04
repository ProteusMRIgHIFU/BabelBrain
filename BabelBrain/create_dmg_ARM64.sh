# Since create-dmg does not clobber, be sure to delete previous DMG
[[ -f BabelBrain_ARM64.dmg ]] && rm BabelBrain_ARM64.dmg
rm -rf final
mkdir final
mkdir final/Profiles
mkdir final/PlanningModels
cp -r ../Profiles/* final/Profiles
cp -r ../PlanningModels/* final/PlanningModels
cp -r ../LICENSE final/
cp BabelBrain_ARM64.pkg final/
# Create the DMG   --hide-extension "BabelBrain.app" \  --app-drop-link 600 185 \
create-dmg \
  --volname "BabelBrain Installer" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "BabelBrain_ARM64.pkg" 200 190 \
  "BabelBrain_ARM64.dmg" \
  "final/"
