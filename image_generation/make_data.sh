i=0
while [ "$i" -le 200 ]; do
    /common/home/gs790/blender/blender --background --python render_images.py -- --num_images 100
#    echo 'hello'
    i=$(( i + 1 ))
done