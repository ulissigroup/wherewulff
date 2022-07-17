# Get all the POSCARS in the sub-folders


if [ ! -e ./IrO2_workflow_jh ]; then
  mkdir ./IrO2_workflow_jh;
fi
find . -name "*inputs*" -type d -print | grep -v pbx | grep -v bulk | grep -v slab |
  while read -r file_name; do
    #echo $file_name;
    name=$(echo $file_name | awk -F '\/' '{print $2""}');
    dir_name=$(echo $file_name | awk -F '\/' '{print $1"/"$2""}');
    #echo ${dir_name};
    echo "cp -r $dir_name into ./IrO2_workflow_jh/${name} dir";
    cp -r $dir_name ./IrO2_workflow_jh/${name};
   done
