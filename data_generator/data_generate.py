import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
#img = cv2.imread('test.jpg')
vid = cv2.VideoCapture(0)
count = 0
while(count < 100):
    count += 1
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print((x,y,w,h))
        saved_img = img[y: y+h, x:x+w, :]
        cv2.imwrite('data/left/{}.jpg'.format(count), saved_img)
    cv2.imshow('img', img)
# Display the output
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
#vid.release()
# Destroy all the windows
#cv2.destroyAllWindows()
# Convert into grayscale
# Detect faces
# Draw rectangle around the faces
# Display the output
cv2.waitKey()

