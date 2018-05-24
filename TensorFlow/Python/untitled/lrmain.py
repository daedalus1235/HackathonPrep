from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import era

def main(argv):
    assert len(argv) == 1

    (train, test) = era.dataset()

    # Switch the labels to units of thousands for better convergence.
    # def to_thousands(features, labels):
    #    return features, labels

    # train = train.map(to_thousands)
    # test = test.map(to_thousands)

    # Build the training input_fn.
    def input_train():
        return (
            # Shuffling with a buffer larger than the data set ensures
            # that the examples are well mixed.
            train.shuffle(1000).batch(16)
                # Repeat forever
                .repeat().make_one_shot_iterator().get_next())

    # Build the validation input_fn.
    def input_test():
        return (test.shuffle(1000).batch(16)
                .make_one_shot_iterator().get_next())

    feature_columns = [
        # era is a  numeric columns
        tf.feature_column.numeric_column(key="era"),
    ]

    # Build the Estimator.
    model = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                         optimizer='Adam')

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    model.train(input_fn=input_train, steps = 1300)  # 1300 steps

    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=input_test)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    print("\nRMS error for the test set: {:.0f}"
          .format(average_loss ** 0.5))

    # Run the model in prediction mode.
    input_dict = {
        "era": np.array([0, 5.36]),
    }
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        input_dict, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)

    actual = [0,64]

    # Print the prediction results.
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        msg = ("ERA: {: 1.4f}, "
               "Wins: {: 4.2f}, ")
        msg = msg.format(input_dict["era"][i],
                         prediction["predictions"][0]
                         )

        print("    " + msg)
    print()


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
