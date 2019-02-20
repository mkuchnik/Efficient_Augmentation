import sklearn

def flatten_batch_X(X):
    """ Flattens images """
    return X.reshape(len(X), -1)


def div_scaler(X, divider):
    """ Divides the input by the divider """
    return X.astype("float32") / divider


def sub_scaler(X, sub):
    """ Subtracts the input by the sub """
    return X - sub


def flattener_transformer():
    """ This flattens images before going into the classifier """
    transformer = sklearn.preprocessing.FunctionTransformer(
        func=flatten_batch_X,
        validate=False,
    )
    return transformer


def image_rescaler_transformer():
    """ This divides images by 255 to convert to [0, 1] range """
    transformer = sklearn.preprocessing.FunctionTransformer(
        func=div_scaler,
        validate=False,
        kw_args={"divider": 255.0},
    )
    return transformer


def image_rescaler_sub_transformer(x_train_mean):
    """ Substracts what's usually the mean """
    transformer = sklearn.preprocessing.FunctionTransformer(
        func=sub_scaler,
        validate=False,
        kw_args={"sub": x_train_mean},
    )
    return transformer
