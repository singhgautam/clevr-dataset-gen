i=0
while [ "$i" -le 40 ]; do
    /common/home/gs790/blender-2-78/blender --background --python render_images.py -- --num_images 100
    i=$(( i + 1 ))
done