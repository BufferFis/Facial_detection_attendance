import os
import cv2
import helper as hp
from flask_session import Session
from flask import Flask, session, request, render_template, redirect, flash

# Defining Flask App
app = Flask(__name__)

# Handling session requests and preventing caching user data across sessions.
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

nimgs = 10

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = hp.extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=hp.totalreg(), datetoday2=hp.datetoday2)

# Example usage of verification
@app.route('/verify')
def verify():
    # Check user is verified
    if not session.get("verified", False):
        # if not verified redirect to homepage with an appropriate message
        flash("User not verified!")
        return redirect('/')
    # if verified then move on to the webpage (Here I have redirected to google for testing purposes)
    return redirect('https://google.com')

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = hp.getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=hp.totalreg(), datetoday2=hp.datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    hp.deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        hp.train_model()
    except:
        pass

    userlist, names, rolls, l = hp.getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=hp.totalreg(), datetoday2=hp.datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = hp.extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=hp.totalreg(), datetoday2=hp.datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(hp.extract_faces(frame)) > 0:
            (x, y, w, h) = hp.extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = hp.identify_face(face.reshape(1, -1))[0]
            if identified_person:
                session["verified"] = True

            hp.add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = hp.extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=hp.totalreg(), datetoday2=hp.datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        faces = hp.extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    hp.train_model()
    names, rolls, times, l = hp.extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=hp.totalreg(), datetoday2=hp.datetoday2)


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
