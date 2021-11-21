# Import Required Libraries
import tensorflow as tf
import numpy as np
import cv2


class DetectKeypoints:

    # Constructor - To initialize objects with values
    def __init__(self, path, Threshold_Value, output_path, interpreter, EDGES):
        self.path = path
        self.Threshold_Value = Threshold_Value
        self.output_path = output_path
        self.interpreter = interpreter
        self.EDGES = EDGES


    # Draw Keypoints
    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        y, x, c = frame.shape # frame.shape = (480,640,3)
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for kp in shaped:
            ky, kx, k_conf = kp
            
            if k_conf > confidence_threshold:
                cv2.circle(frame, (round(kx),round(ky)), 4, (255,0,0), -1)


    # Draw Connections
    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape # frame.shape = (480,640,3)
        shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))
        
        for edge,color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (round(x1),round(y1)), (round(x2),round(y2)), (0,255,0), 2)


    # Make Detection
    def make_detection(self):
        # WebCam/Video (If path=0 it is WebCam and If path='/File Path' it is a video)
        cap = cv2.VideoCapture(self.path)

        # Used for Downloading Video
        if self.output_path != None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (width,height))

        while cap.isOpened():
            # Read from WebCam/Video by frames
            ret,frame = cap.read()

            if ret == True:        
                # Reshape Image - Because MoveNet is accepting input in 192x192x3 and data type must be float32
                img = frame.copy()
                img = tf.image.resize_with_pad(np.expand_dims(img,axis=0), 192, 192)
                input_image = tf.cast(img, dtype=tf.float32)
                
                
                # Set Input and Output
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                # Make Predictions
                self.interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
                self.interpreter.invoke()
                keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
                print(keypoints_with_scores)
                
                # Draw Connections
                self.draw_connections(frame, keypoints_with_scores, self.EDGES, self.Threshold_Value)
                
                # Draw Keypoints
                self.draw_keypoints(frame, keypoints_with_scores, self.Threshold_Value)

                # Download Video
                if self.output_path != None:
                    out.write(frame)
                
                # Show WebCam/Video
                cv2.imshow('MoveNet Lightning', frame)
                
                # Stop the WebCam/Video when 'q' is pressed on the keyboard
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            else:
                break

        # Release the WebCam/Video
        cap.release()

        # Release Storing Video
        if self.output_path != None:
            out.release()

        # Destroy all windows
        cv2.destroyAllWindows()


    # Destructor - To delete all objects before code ends
    def __del__(self):
        print('Objects Deleted')


# -------------------------------------------------------------------------------END---------------------------------------------------------------------------------------