i=0
while [ "$i" -le 2 ]; do
    /common/home/gs790/blender-2-91/blender --background --python render_images.py -- --num_images 100
    i=$(( i + 1 ))
done