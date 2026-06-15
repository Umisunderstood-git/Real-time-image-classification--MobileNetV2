import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input, decode_predictions


def ImageClassifier():
    cap=cv2.VideoCapture(0)
    if cap.isOpened()==False:
        print("Faced issues while accessing the camera")
        return
    else:
        print("Opened camera")
    
    while True:
        ret,frame=cap.read() #ret gives boolean value, False if frames are not detected
        
        if not ret:
            print("Error: Could not read frame")
            break
        

        #Preprocessing for our MobileNetV2 model: BGR--->RGB (OpenCV stores BGR by default)
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_resized=cv2.resize(img,(224,224))
        ##MobilenetV2 expects image matrix in 4 dimensions---> 1 height,1 width, 1 channel and 1 batch size
        x=np.expand_dims(img_resized,axis=0).astype(np.float32) #expand_dims also incrases the memory size of the array. axis= parameter adds new dimension
        x=preprocess_input(x)



      #Creating a object of pretrained Imagenet Model 
        model=MobileNetV2(weights="imagenet")
    

      #Predict -> Inference:
        preds=model.predict(x,verbose=0)
        decoded=decode_predictions(preds,top=1)[0][0] #(class_id,class_name,score) (It will return this tupple)
        label=f"{decoded[1]}:{decoded[2]*100}%"        
       
        #Now overlay prediction on the camera frame:
        cv2.putText(frame,label,(16,40),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Mandars-Real time image classification using- MobileVnet2 CNN",frame)

    # 6) Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    #Cleanup
    cap.release()
    cv2.destroyAllWindows()





def main():
    ImageClassifier()



if __name__=="__main__":
    ImageClassifier()
    
        