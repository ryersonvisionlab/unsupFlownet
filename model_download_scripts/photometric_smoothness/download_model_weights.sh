# download weights
echo "Downloading weights..." 
#wget -O weights.tar.gz https://doc-0k-5o-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/1q5snsmj588h2s8eh09lc9d6qiltvspu/1538179200000/05082393995079092114/*/18MXPbjXUnGG_Fcij8YUs6mtyQcBwnSru?e=download
echo "Please download this file and place it in this directory: https://drive.google.com/file/d/18MXPbjXUnGG_Fcij8YUs6mtyQcBwnSru/view?usp=sharing"
read -p "Press to continue once weights.tar.gz is in this directory"
# ask if we want to automatically copy weights and params into src
while true; do
    read -p "Do you want the weights and hyper-parameters copied to src? (will overwrite)" yn
    case $yn in
        [Yy]* ) echo "Extracting and copying...";tar -xzvf weights.tar.gz;cp weights/* ../../src/snapshots/.;cp hyperParams.json ../../src/.; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer y or n.";;
    esac
done
