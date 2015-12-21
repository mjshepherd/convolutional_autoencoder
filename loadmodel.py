import cPickle
import plotutils as charts
import sys
from loaddata import read_dataset
import utils
import numpy as np

if __name__ == '__main__':
    test_set = read_dataset('small_test_10')
    f = file(sys.argv[1], 'rb')
    loaded_model = cPickle.load(f)
    f.close()

    test_image = test_set
    cor_test_image = utils.create_random_hole(test_image)
    recon = [loaded_model['predict'](np.expand_dims(image, axis=0))[0] for image in cor_test_image]

    stats = loaded_model['stats']
    charts.show_weights(loaded_model['weights'])
    charts.ShowImages([test_image, cor_test_image, recon])
    charts.plot_error(stats)
