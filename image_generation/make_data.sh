i=0
while [ "$i" -le 2 ]; do
    /common/home/gs790/blender-3-00/blender --background --python render_images.py -- --num_images 100
    i=$(( i + 1 ))
done