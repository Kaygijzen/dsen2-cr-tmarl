import keras.backend as K
import tensorflow as tf


def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return K.mean(K.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return K.mean(K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]), axis=[2, 3])))


def cloud_ssim(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = y_true[:, 0:13, :, :]
    y_pred = y_pred[:, 0:13, :, :]

    y_true *= 2000
    y_pred *= 2000

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    ssim = tf.image.ssim(y_true, y_pred, max_val=10000.0)
    ssim = tf.reduce_mean(ssim)

    return ssim


def get_sam(y_true, y_predict):
    """Computes the SAM array."""
    mat = tf.multiply(y_true, y_predict)
    mat = tf.reduce_sum(mat, 1)
    mat = tf.div(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), 1)))
    mat = tf.div(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_predict, y_predict), 1)))
    mat = tf.acos(K.clip(mat, -1, 1))

    return mat


def cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image."""
    mat = get_sam(y_true[:, 0:13, :, :], y_predict[:, 0:13, :, :])

    return tf.reduce_mean(mat)


def cloud_mean_sam_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    target = y_true[:, 0:13, :, :]
    predicted = y_pred[:, 0:13, :, :]

    if K.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    sam = get_sam(target, predicted)
    sam = tf.expand_dims(sam, 1)
    sam = K.sum(cloud_cloudshadow_mask * sam) / K.sum(cloud_cloudshadow_mask)

    return sam


def cloud_mean_sam_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

    if K.sum(clearmask) == 0:
        return 0.0

    sam = get_sam(input_cloudy, predicted)
    sam = tf.expand_dims(sam, 1)
    sam = K.sum(clearmask * sam) / K.sum(clearmask)

    return sam


def cloud_psnr(y_true, y_predict):
    """Computes the PSNR over the full image."""
    y_true *= 2000
    y_predict *= 2000
    rmse = K.sqrt(K.mean(K.square(y_predict[:, 0:13, :, :] - y_true[:, 0:13, :, :])))

    return 20.0 * (K.log(10000.0 / rmse) / K.log(10.0))


def cloud_mean_absolute_error_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

    if K.sum(clearmask) == 0:
        return 0.0

    clti = clearmask * K.abs(predicted - input_cloudy)
    clti = K.sum(clti) / (K.sum(clearmask) * 13)

    return clti


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    target = y_true[:, 0:13, :, :]

    if K.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    ccmaec = cloud_cloudshadow_mask * K.abs(predicted - target)
    ccmaec = K.sum(ccmaec) / (K.sum(cloud_cloudshadow_mask) * 13)

    return ccmaec


def carl_error(y_true, y_pred):
    """Computes the Cloud-Adaptive Regularized Loss (CARL)"""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]
    target = y_true[:, 0:13, :, :]

    cscmae = K.mean(clearmask * K.abs(predicted - input_cloudy) + cloud_cloudshadow_mask * K.abs(
        predicted - target)) + 1.0 * K.mean(K.abs(predicted - target))

    return cscmae


def tmarl_error(y_true, y_pred, alpha=0.5, r_lambda=1): # CONTRIBUTION: the parameterized TMARL function
    """Computes the Twin Mask Adaptive Regularized Loss (TMARL)"""
    print("TMARL: using alpha =", alpha, "and lambda =", r_lambda)
    
    # y_true = [
    #   0:13 sentinel-2 channels
    #   14 cloud mask (CM)
    #   15 cloud-shadow mask (SM)
    #   16 merged cloud and cloud-shadow mask (CSM)
    # ]
    
    cloud_mask = y_true[:, -3:-2, :, :] # third last entry contains the cloud mask
    shadow_mask = y_true[:, -2:-1, :, :] # second last entry contains the cloud-shadow mask

    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]
    target = y_true[:, 0:13, :, :]

    # generate cloud-free and shadow-free masks
    clearmask_cloud = K.ones_like(cloud_mask) - cloud_mask
    clearmask_shadow = K.ones_like(shadow_mask) - shadow_mask

    # compute the errors
    error_cloud = K.mean(clearmask_cloud * K.abs(predicted - input_cloudy) + cloud_mask * K.abs(
           predicted - target))
    error_shadow = K.mean(clearmask_shadow * K.abs(predicted - input_cloudy) + shadow_mask * K.abs(
            predicted - target))
    error_reg = 1.0 * K.mean(K.abs(predicted - target))

    # parameterized TMARL function
    tmarl = alpha * error_cloud + (1 - alpha) * error_shadow + r_lambda * error_reg

    return tmarl
