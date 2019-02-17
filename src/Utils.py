import numpy as np
                 
def normalize_data(data, mu=None, sigma=None, epsilon=10e-8):
    epsilon = 10e-8
    if mu is None:
        mu = np.mean(data, axis=0)
    if sigma is None:
        sigma = np.sqrt(np.var(data, axis=0))
    norm_data = (data-mu)/(sigma+epsilon)
    return norm_data, mu, sigma

def unnormalize_data(norm_data, mu, sigma, epsilon=10e-8):
    data = norm_data*(sigma+epsilon) + mu
    return data
        
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label.reshape((true_label.shape[0],))[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label.reshape((true_label.shape[0],))[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')