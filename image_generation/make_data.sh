i=0
while [ "$i" -le 300 ]; do
    /common/home/gs790/blender-2-91-v2/blender --background --python render_images.py -- --num_images 100
    i=$(( i + 1 ))
done