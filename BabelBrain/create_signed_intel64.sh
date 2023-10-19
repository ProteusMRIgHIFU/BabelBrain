#script to run all steps to create app, package , dmg , codesign and notarize
rm -rf dist build 
pyinstaller BabelBrain.spec
python pycodesign.py -s -p  -n -t -v "/Users/spichardo/Library/Mobile Documents/com~apple~CloudDocs/pycodesign/pycodesign_intel64.ini"
source create_dmg_Intel64.sh
rm -rf dist build
source "/Users/spichardo/Library/Mobile Documents/com~apple~CloudDocs/pycodesign/notarize_dmg_intel64.sh"
xcrun stapler staple BabelBrain_Intel64.dmg
