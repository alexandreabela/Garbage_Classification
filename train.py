from params import *

def train_model(model, train_data_generator, validation_data_generator): 
        ## Compilation du modèle 
        model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])

        ## Entraînement du modèle 
        model.fit(
                train_data_generator,
                validation_data = validation_data_generator,
                epochs=no_epochs
                )

        #loss_train=history.history["loss"]
        #metrics_train=history.history["accuracy"]

        #loss_test=history.history["val_loss"]
        #metrics_test=history.history["val_accuracy"]

        #plt.figure()
        #plt.title("loss ")
        #plt.plot(epochs,loss_train,'r+',label="Training loss")
        #plt.plot(epochs,loss_test,'b+',label="Testing loss")
        #plt.legend()
        #plt.xlabel("epochs")
        #plt.show()

        #plt.figure()
        #plt.title("accuracy ")
        #plt.plot(epochs,metrics_train,'r+',label="Training accuracy")
        #plt.plot(epochs,metrics_test,'b+',label="Testing accuracy")
        #plt.legend()
        #plt.xlabel("epochs")
        #plt.show()
        return history