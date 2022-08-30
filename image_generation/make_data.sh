i=0
while [ "$i" -le 300 ]; do
    /common/home/gs790/blender/blender --background --python render_images.py -- --num_images 100
    i=$(( i + 1 ))
done