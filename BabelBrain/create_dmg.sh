# Since create-dmg does not clobber, be sure to delete previous DMG
[[ -f BabelBrain.dmg ]] && rm BabelBrain.dmg
[[ -d dist/BabelBrain ]] && rm -rf dist/BabelBrain
[[ -d dist/ThermalProfiles ]] && rm -rf dist/ThermalProfiles
[[ -d dist/PlanningModels ]] && rm -rf dist/PlanningModels

cp -r ../ThermalProfiles dist/
cp -r ../PlanningModels dist/

# Create the DMG
create-dmg \
  --volname "BabelBrain Installer" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "BabelBrain.app" 200 190 \
  --hide-extension "BabelBrain.app" \
  --app-drop-link 600 185 \
  "BabelBrain.dmg" \
  "dist/"
