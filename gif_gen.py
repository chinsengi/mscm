import imageio
import os
import argparse

parser = argparse.ArgumentParser('gif_gen')
parser.add_argument('--alp', type=float, default=0.005)
parser.add_argument('--save', type=str, default='./experiments')
args = parser.parse_args()

images = []
filenames = [os.path.join(args.save, 'gif_{}\generate-itr-{}.jpg'.format(args.alp,i)) for i in range(4, 200,4)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('{}-dynamics.gif'.format(args.alp), images, 'GIF') 