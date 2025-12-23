import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """
        Load the pre-trained base model and compile it using legacy SGD optimizer.
        """
        self.model = tf.keras.models.load_model(str(self.config.updated_base_model_path), compile=False)

        # Compile manually using legacy SGD (for M1/M4 Macs)
        from tensorflow.keras.optimizers import legacy
        self.model.compile(
            optimizer=legacy.SGD(learning_rate=0.01, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """
        Create training and validation generators with optional data augmentation.
        """
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the model in HDF5 format to avoid 'options' error.
        """
        filepath = str(path)
        if not filepath.endswith(".h5"):
            filepath += ".h5"
        model.save(filepath, save_format="h5")

    def train(self, callback_list: list):
        """
        Train the model using the training and validation generators.
        """
        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
