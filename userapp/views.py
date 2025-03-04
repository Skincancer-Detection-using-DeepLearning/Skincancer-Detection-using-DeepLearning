from django.shortcuts import render,redirect
from django.contrib import messages
import time
import pandas as pd
from userapp.models import *
from adminapp.models import *
from mainapp.models import *
import pytz
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from userapp.views import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib
from django.core.files.storage import default_storage
from django.conf import settings
from django.core.paginator import Paginator
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import cv2
import base64
import io
# Create your views here.
def user_dashboard(req):
    prediction_count = UserModel.objects.all().count()
    print(prediction_count)
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id = user_id)
    Feedbacks_users_count= Feedback.objects.all().count()
    all_users_count =  UserModel.objects.all().count()
    if user.Last_Login_Time is None:
        IST = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(IST).time()
        user.Last_Login_Time = current_time_ist
        user.save()
        return redirect('user_dashboard')
    return render(req,'user/user-dashboard.html', {'predictions' : prediction_count, 'la' : user,'a':Feedbacks_users_count,'a':all_users_count})



def user_profile(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id = user_id)
    if req.method == 'POST':
        user_name = req.POST.get('username')
        user_age = req.POST.get('age')
        user_phone = req.POST.get('mobile number')
        user_email = req.POST.get('email')
        user_password = req.POST.get('Password')
        user_address = req.POST.get("address")
        
        #user_img = req.POST.get("userimg")

        user.user_name = user_name
        user.user_age = user_age
        user.user_address = user_address
        user.user_contact = user_phone
        user.user_email=user_email
        user.user_password=user_password
       
       

        if len(req.FILES) != 0:
            image = req.FILES['profilepic']
            user.user_image = image
            user.user_name = user_name
            user.user_age = user_age
            user.user_contact = user_phone
            user.user_email=user_email
            user.user_address = user_address
            user.user_password = user_password
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
        else:
            user.user_name = user_name
            user.user_age = user_age
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
            
    context = {"i":user}
    return render(req,'user/user-profile.html',context)


# Video camera class to capture frames
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # Open the default camera

    def __del__(self):
        self.video.release()  # Release the camera when done

    def get_frame(self):
        success, image = self.video.read()  # Read a frame from the camera
        if not success:
            return None
        ret, jpeg = cv2.imencode('.jpg', image)  # Encode the frame as JPEG
        return jpeg.tobytes()  # Return the byte data of the frame
from django.http import StreamingHttpResponse

# View to stream video feed from the webcam
def webcam_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

# Generator function to yield frames from the camera
def gen(camera):
    while True:
        frame = camera.get_frame()  # Get a frame from the camera
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
from .models import Skin_cancer_dataset

def predict_image_class(image_path):
    # Load the pre-trained EfficientNetB0 model
    model = load_model('inceptionv3.hdf5')

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    return np.argmax(predictions)

def Classification(req):
    if req.method == 'POST' and req.FILES['image']:
        uploaded_file = req.FILES['image']
        model_type = req.POST.get('model_type')
        
        # Save the uploaded image using default storage
        image_name = default_storage.save(uploaded_file.name, uploaded_file)
        image_path = default_storage.path(image_name)
        imgs = img(image_path)

            # Encode images to base64
        with open(image_path, "rb") as img_file:
                uploaded_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        with open(imgs, "rb") as img_file:
                simg = base64.b64encode(img_file.read()).decode('utf-8')
        # Perform classification
        predicted_label = predict_image_class(default_storage.path(image_name))

        # Map predicted label to disease type
        disease_types = {
            0: 'Actinic keratoses',
            1: 'Basal cell carcinoma',
            2: 'Benign keratosis',
            3: 'Dermatofibroma',
            4: 'Melanoma',
            5: 'Melanocytic nevi',
            6: 'Vascular lesions'
        }
        predicted_disease = disease_types.get(predicted_label, 'Unknown')
        print(predicted_disease)

        if model_type == 'U-Net++':
            model_info = All_model.objects.get(model_Name='U-Net++')
        else:
            # Handle invalid model type
            model_info = None
        print(model_info,"asdsasadasdassdsasd")
        # Save the image path, predicted disease, and accuracy to the session
        req.session["uploaded_image_base64"] = uploaded_image_base64
        req.session["segmented_image_base64"] = simg
        req.session['image_path'] = image_path
        req.session['predicted_disease'] = predicted_disease

        if model_info:
            req.session['model_name'] = model_info.model_Name
            req.session['model_accuracy'] = model_info.model_accuracy

        # Redirect to Classification_result view
        return redirect('Classification_result')
    else:
        # Render the Classification template
        return render(req, 'user/classification.html')





def img(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    im = os.path.splitext(image_path)[0] + '_segmented.jpg'
    cv2.imwrite(im, binary_image)
    
    return im



import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import CustomObjectScope

# Define custom metrics if they were used in the model
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Function to load the model and predict the result
def predict_skin_lesion(image_path, model_path="skindoubleunet.h5"):
    # Load the saved model
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.resize(image, (256, 256))
    image = original_image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict the mask
    prediction = model.predict(image)[0] > 0.5
    prediction = np.squeeze(prediction, axis=-1)
    prediction = prediction.astype(np.float32)
    
    # Display the original image and the predicted mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image[..., ::-1])  # Convert BGR to RGB
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    
    plt.show()

    return prediction

# Example usage:


def Classification_unet(req):
    if req.method == 'POST' and req.FILES['image']:
        uploaded_file = req.FILES['image']
        model_type = req.POST.get('model_type')
        
        # Save the uploaded image using default storage
        image_name = default_storage.save(uploaded_file.name, uploaded_file)
        image_path = default_storage.path(image_name)
        imgs = img(image_path)

            # Encode images to base64
        with open(image_path, "rb") as img_file:
                uploaded_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        with open(imgs, "rb") as img_file:
                simg = base64.b64encode(img_file.read()).decode('utf-8')
        # Perform classification
        predicted_label = predict_skin_lesion(default_storage.path(image_name))

    return render(req,'user/Classificationu-net.html')


from django.http import HttpResponse

def Classification_result(req):
    # Retrieve image path, predicted disease, and accuracy from session
    image_path = req.session.get('uploaded_image_base64', None)
    predicted_disease = req.session.get('predicted_disease', 'Unknown')
    model_accuracy = req.session.get('model_accuracy', None)
    model_name = req.session.get('model_name', None)
    img = req.session.get("segmented_image_base64", None)
    uploaded_image_base64 = req.session.get("uploaded_image_base64", None)

    # Debugging: Print session values to console
    print("Predicted Disease in Classification Result View:", predicted_disease)
    print("Model Accuracy in Classification Result View:", model_accuracy)
    print("Model Name in Classification Result View:", model_name)

    disease_info = {
    'Actinic keratoses': {
           
            'es': {
                'symptoms': 'Parches ásperos y escamosos en áreas expuestas al sol. Pueden sentirse sensibles o con picazón. Aparecen en cara, orejas, cuero cabelludo, hombros y cuello.',
                'medication': 'Tratamientos tópicos: crema de 5-fluorouracilo, Imiquimod. Procedimientos: Crioterapia, Terapia fotodinámica, Ablación láser.'
            },
            'fr': {
                'symptoms': 'Plaques rugueuses et squameuses sur les zones exposées au soleil. Peuvent être sensibles ou démanger. Apparaissent sur le visage, les oreilles, le cuir chevelu, les épaules et le cou.',
                'medication': 'Traitements topiques : crème au 5-fluorouracile, Imiquimod. Procédures : Cryothérapie, Thérapie photodynamique, Ablation au laser.'
            },
            'de': {
                'symptoms': 'Raue, schuppige Flecken auf sonnenexponierten Stellen. Können empfindlich oder juckend sein. Treten im Gesicht, an Ohren, Kopfhaut, Schultern und Nacken auf.',
                'medication': 'Topische Behandlungen: 5-Fluorouracil-Creme, Imiquimod. Verfahren: Kryotherapie, Photodynamische Therapie, Laserablation.'
            },
            'hi': {
                'symptoms': 'धूप वाले क्षेत्रों में खुरदुरे, पपड़ीदार धब्बे। कोमल या खुजली हो सकती है। चेहरे, कान, गंजे सिर, कंधों और गर्दन पर दिखाई देते हैं।',
                'medication': 'टॉपिकल उपचार: 5-फ्लोरोयूरासिल क्रीम, इमीक्विमोड। प्रक्रियाएं: क्रायोथेरेपी, फोटोडायनामिक थेरेपी, लेजर एब्लेशन।'
            }
        },

     'Basal cell carcinoma': {
    'en': {
        'symptoms': 'Pearly bump with visible blood vessels. May ulcerate or bleed easily. Slow-growing but can be destructive.',
        'medication': 'Surgical excision, Mohs surgery, Electrodesiccation, Radiation therapy for advanced cases.'
    },
    'es': {
        'symptoms': 'Bulto nacarado con vasos sanguíneos visibles. Puede ulcerarse o sangrar fácilmente. Crecimiento lento pero destructivo.',
        'medication': 'Escisión quirúrgica, Cirugía de Mohs, Electrodesecación, Radioterapia para casos avanzados.'
    },
    'fr': {
        'symptoms': 'Bosse nacrée avec vaisseaux sanguins visibles. Peut sulcérer ou saigner facilement. Croissance lente mais destructive.',
        'medication': 'Excision chirurgicale, Chirurgie de Mohs, Électrodessiccation, Radiothérapie pour cas avancés.'
    },
    'de': {
        'symptoms': 'Perliger Knoten mit sichtbaren Blutgefäßen. Kann ulzerieren oder leicht bluten. Langsam wachsend aber destruktiv.',
        'medication': 'Chirurgische Exzision, Mohs-Chirurgie, Elektrodesikkation, Strahlentherapie bei fortgeschrittenen Fällen.'
    },
    'hi': {
        'symptoms': 'दिखाई देने वाली रक्त वाहिकाओं वाला मोती जैसा उभार। आसानी से छाले या रक्तस्राव हो सकता है। धीमी गति से बढ़ने वाला लेकिन विनाशकारी।',
        'medication': 'सर्जिकल एक्सिशन, मोह्स सर्जरी, इलेक्ट्रोडिसिकेशन, उन्नत मामलों के लिए रेडिएशन थेरेपी।'
    }
},
# Continue similar structure for all other diseases...
    'Benign keratosis': {
        'en': {
            'symptoms': 'Warty or waxy growths, typically brown/black/tan. Usually appears on face, chest, or back. Non-cancerous but may resemble skin cancer.',
            'medication': 'Cryotherapy, Curettage with electrodesiccation, Laser ablation, Topical retinoids for cosmetic improvement.'
        },
        'es': {
            'symptoms': 'Crecimientos verrugosos o cerosos, normalmente marrones/negros/beige. Aparecen en cara, pecho o espalda. No cancerosos pero similares a cáncer de piel.',
            'medication': 'Crioterapia, Legrado con electrodesecación, Ablación láser, Retinoides tópicos para mejoría cosmética.'
        },
        'fr': {
            'symptoms': 'Excroissances verruqueuses ou cireuses, généralement brunes/noires/beiges. Apparaissent sur le visage, la poitrine ou le dos. Non cancéreuses mais ressemblant au cancer de la peau.',
            'medication': 'Cryothérapie, Curetage avec électrodessiccation, Ablation laser, Rétinoïdes topiques pour amélioration cosmétique.'
        },
        'de': {
            'symptoms': 'Warzenartige oder wachsige Wucherungen, typischerweise braun/schwarz/beige. Erscheinen im Gesicht, auf der Brust oder dem Rücken. Gutartig, aber hautkrebsähnlich.',
            'medication': 'Kryotherapie, Kürettage mit Elektrodesikkation, Laserablation, Topische Retinoide zur kosmetischen Verbesserung.'
        },
        'hi': {
            'symptoms': 'मस्सेदार या मोम जैसे उभार, आमतौर पर भूरे/काले/टैन रंग के। चेहरे, छाती या पीठ पर दिखाई देते हैं। कैंसर रहित लेकिन त्वचा कैंसर जैसे दिख सकते हैं।',
            'medication': 'क्रायोथेरेपी, इलेक्ट्रोडिसिकेशन के साथ क्यूरेटेज, लेजर एब्लेशन, कॉस्मेटिक सुधार के लिए टॉपिकल रेटिनोइड्स।'
        }
    },

    
    'Dermatofibroma': {
        'en': {
            'symptoms': 'Small, firm bump often with dimple sign. Typically reddish-brown. Commonly appears on legs. May itch or be sensitive to touch.',
            'medication': 'Observation recommended, Surgical excision if symptomatic, Laser removal for cosmetic purposes.'
        },
        'es': {
            'symptoms': 'Bulto pequeño y firme con hoyuelo. Color rojizo-marrón. Común en piernas. Puede picar o ser sensible al tacto.',
            'medication': 'Observación recomendada, Extirpación quirúrgica si hay síntomas, Eliminación láser con fines cosméticos.'
        },
        'fr': {
            'symptoms': 'Petite bosse ferme avec signe de fossette. Généralement rouge-brun. Apparaît souvent sur les jambes. Peut démanger ou être sensible au toucher.',
            'medication': 'Surveillance recommandée, Excision chirurgicale si symptomatique, Ablation laser pour raisons esthétiques.'
        },
        'de': {
            'symptoms': 'Kleine, feste Beule oft mit Dellenzeichen. Rötlich-braun. Erscheint häufig an den Beinen. Kann jucken oder berührungsempfindlich sein.',
            'medication': 'Beobachtung empfohlen, Chirurgische Entfernung bei Symptomen, Laserentfernung aus kosmetischen Gründen.'
        },
        'hi': {
            'symptoms': 'छोटा, सख्त उभार जिसमें डिंपल हो सकता है। लाल-भूरा रंग। आमतौर पर पैरों पर दिखता है। खुजली या स्पर्श के प्रति संवेदनशील हो सकता है।',
            'medication': 'निरीक्षण की सलाह, लक्षण होने पर सर्जिकल निष्कासन, कॉस्मेटिक उद्देश्य के लिए लेजर निष्कासन।'
        }
    },
    'Melanoma': {
        'en': {
            'symptoms': 'Asymmetric mole with irregular borders and color variations. May change size, shape, or texture. Can occur anywhere on body.',
            'medication': 'Wide local excision, Sentinel lymph node biopsy, Immunotherapy, Targeted therapy for advanced cases.'
        },
        'es': {
            'symptoms': 'Lunar asimétrico con bordes irregulares y variaciones de color. Puede cambiar tamaño, forma o textura. Puede aparecer en cualquier parte del cuerpo.',
            'medication': 'Escisión amplia, Biopsia de ganglio centinela, Inmunoterapia, Terapia dirigida para casos avanzados.'
        },
        'fr': {
            'symptoms': "Grain de beauté asymétrique aux bords irréguliers et couleurs variées. Peut changer de taille, forme ou texture. Peut apparaître n'importe où sur le corps.",
            'medication': 'Excision large, Biopsie du ganglion sentinelle, Immunothérapie, Thérapie ciblée pour cas avancés.'
        },
        'de': {
            'symptoms': 'Asymmetrischer Leberfleck mit unregelmäßigen Rändern und Farbvariationen. Kann Größe, Form oder Textur ändern. Kann überall am Körper auftreten.',
            'medication': 'Weiträumige Exzision, Wächterlymphknotenbiopsie, Immuntherapie, Zielgerichtete Therapie bei fortgeschrittenen Fällen.'
        },
        'hi': {
            'symptoms': 'असममित तिल जिसमें अनियमित किनारे और रंग भिन्नताएं होती हैं। आकार, आकृति या बनावट बदल सकता है। शरीर के किसी भी हिस्से में हो सकता है।',
            'medication': 'वाइड लोकल एक्सिशन, सेंटिनल लिम्फ नोड बायोप्सी, इम्यूनोथेरेपी, उन्नत मामलों के लिए टार्गेटेड थेरेपी।'
        }
    },
    'Melanocytic nevi': {
        'en': {
            'symptoms': 'Common moles, usually round/oval with even color. May be flat or raised. Typically <6mm diameter. Monitor for ABCDE changes.',
            'medication': 'Regular self-exams, Dermoscopic monitoring, Excision if suspicious changes occur.'
        },
        'es': {
            'symptoms': 'Lunares comunes, redondos/ovalados con color uniforme. Planos o elevados. Típicamente <6mm de diámetro. Monitorear cambios ABCDE.',
            'medication': 'Autoexámenes regulares, Monitoreo dermatoscópico, Extirpación si hay cambios sospechosos.'
        },
        'fr': {
            'symptoms': "Grains de beauté communs, ronds/ovales avec couleur uniforme. Plats ou surélevés. Typiquement <6mm de diamètre. Surveiller les changements ABCDE.",
            'medication': 'Auto-examens réguliers, Surveillance dermatoscopique, Excision en cas de changements suspects.'
        },
        'de': {
            'symptoms': 'Häufige Muttermale, rund/oval mit gleichmäßiger Farbe. Flach oder erhaben. Typisch <6mm Durchmesser. Auf ABCDE-Veränderungen achten.',
            'medication': 'Regelmäßige Selbstuntersuchungen, Dermatoskopische Überwachung, Entfernung bei verdächtigen Veränderungen.'
        },
        'hi': {
            'symptoms': 'सामान्य तिल, आमतौर पर गोल/अंडाकार और समान रंग। सपाट या उभरे हुए। आमतौर पर <6mm व्यास। ABCDE परिवर्तनों की निगरानी करें।',
            'medication': 'नियमित स्व-परीक्षा, डर्मोस्कोपिक निगरानी, संदिग्ध परिवर्तन होने पर निष्कासन।'
        }
    },
    'Vascular lesions': {
        'en': {
            'symptoms': 'Red/purple skin marks from blood vessel abnormalities. Includes hemangiomas, port-wine stains, cherry angiomas. May be present at birth or develop later.',
            'medication': 'Pulsed dye laser for red lesions, Sclerotherapy for veins, Observation for asymptomatic cases.'
        },
        'es': {
            'symptoms': 'Marcas rojas/moradas por anomalías vasculares. Incluyen hemangiomas, manchas vino de oporto, angiomas cereza. Pueden estar presentes al nacer o desarrollarse después.',
            'medication': 'Láser de colorante pulsado para lesiones rojas, Escleroterapia para venas, Observación en casos asintomáticos.'
        },
        'fr': {
            'symptoms': 'Marques cutanées rouges/violettes dues à des anomalies vasculaires. Hémangiomes, taches de vin, angiomes cerise. Peuvent être présentes à la naissance ou se développer plus tard.',
            'medication': 'Laser à colorant pulsé pour lésions rouges, Sclérothérapie pour veines, Surveillance des cas asymptomatiques.'
        },
        'de': {
            'symptoms': 'Rote/violette Hautmale durch Gefäßanomalien. Hämangiome, Feuermale, Kirschangiome. Können bei Geburt vorhanden sein oder sich später entwickeln.',
            'medication': 'Gepulster Farbstofflaser für rote Läsionen, Sklerotherapie für Venen, Beobachtung bei asymptomatischen Fällen.'
        },
        'hi': {
            'symptoms': 'रक्त वाहिका असामान्यताओं के कारण लाल/बैंगनी त्वचा के निशान। हेमांजियोमास, पोर्ट-वाइन स्टेन, चेरी एंजियोमा शामिल। जन्म के समय मौजूद या बाद में विकसित हो सकते हैं।',
            'medication': 'लाल घावों के लिए पल्स डाई लेजर, नसों के लिए स्क्लेरोथेरेपी, स्पर्शोन्मुख मामलों के लिए निरीक्षण।'
        }
    }
}

    # Get the symptoms and medication for the predicted disease
    disease_details = disease_info.get(predicted_disease, {'symptoms': 'Unknown', 'medication': 'Unknown'})

    # Store symptoms and medication in session
    req.session['symptoms'] = disease_details['symptoms']
    req.session['medication'] = disease_details['medication']
     # Get default (English) details
    default_details = disease_info.get(predicted_disease, {}).get('en', {})
    
    # Store all translations in session
    req.session['disease_translations'] = disease_info.get(predicted_disease, {})
    
    # Check if image_path is available
    if image_path:
        # Pass the image path, predicted disease, accuracy, symptoms, and medication to the template
        return render(req, 'user/classification-result.html', {
            'uploaded_image_base64': uploaded_image_base64,
            'img': img,
            'image_path': image_path,
            'predicted_disease': predicted_disease,
            'model_accuracy': model_accuracy,
            'model_name': model_name,
            'translations': json.dumps(disease_info.get(predicted_disease, {})),
            'symptoms': default_details.get('symptoms', ''),
            'medication': default_details.get('medication', '')
        })
    else:
        # Handle the case where image_path is not available
        return HttpResponse("Error: Image not found")

import json






def Classification_result(req):
    # Retrieve image path, predicted disease, and accuracy from session
    image_path = req.session.get('uploaded_image_base64', None)
    predicted_disease = req.session.get('predicted_disease', 'Unknown')
    model_accuracy = req.session.get('model_accuracy', None)
    model_name = req.session.get('model_name', None)
    img = req.session.get("segmented_image_base64", None)
    uploaded_image_base64 = req.session.get("uploaded_image_base64", None)

    # Debugging: Print session values to console
    print("Predicted Disease in Classification Result View:", predicted_disease)
    print("Model Accuracy in Classification Result View:", model_accuracy)
    print("Model Name in Classification Result View:", model_name)

    disease_info = {'desies cntent':'3233'}

    # Get the symptoms and medication for the predicted disease
    disease_details = disease_info.get(predicted_disease, {'symptoms': 'Unknown', 'medication': 'Unknown'})

    # Store symptoms and medication in session
    req.session['symptoms'] = disease_details['symptoms']
    req.session['medication'] = disease_details['medication']
    hospitals = Hospital.objects.all()

    # Check if image_path is available
    if image_path:
        # Pass the image path, predicted disease, accuracy, symptoms, and medication to the template
        hospitals = Hospital.objects.all()

    return render(req, 'user/classification-result.html', {
        'uploaded_image_base64': uploaded_image_base64,
        'img': img,
        'image_path': image_path,
        'predicted_disease': predicted_disease,
        'model_accuracy': model_accuracy,
        'model_name': model_name,
        'symptoms': disease_details['symptoms'],
        'medication': disease_details['medication'],
        'hospitals': hospitals
    })


def suggest_hospitals(request):
    hospitals = Hospital.objects.all()
    return render(request, 'user/suggest_hospitals.html', {'hospitals': hospitals})











































def user_feedback(req):   
    id=req.session["user_id"]
    uusser=UserModel.objects.get(user_id=id)
    if req.method == "POST":
        rating=req.POST.get("rating")
        review=req.POST.get("review")
        # print(sentiment)        
        # print(rating)
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(review)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment=' very negative'
        else :
            sentiment='neutral'
        Feedback.objects.create(Rating=rating,Review=review,Sentiment=sentiment,Reviewer=uusser)
        messages.success(req,'Feedback recorded')
        return redirect('user_feedback')
    return render(req,'user/user-feedback.html')


def user_logout(req):
    view_id = req.session["user_id"]
    user = UserModel.objects.get(user_id = view_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(req, 'You are logged out..')
    return redirect('user_login')

#--------------------------------------------------------------------------
import tkinter as tk

def update_label(label, text):
    label.config(text=text)

def main():
    root = tk.Tk()
    label = tk.Label(root, text="Initial Text")
    label.pack()

    def update_text():
        # This function is called from the main thread
        update_label(label, "Updated Text")

    # Schedule the update_text function to be called from the main thread
    root.after(1000, update_text)

    root.mainloop()

if __name__ == "__main__":
    main()

#----------------------------------------------------------------
from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def download_classification_data_pdf(req):
    # Retrieve data from session or wherever you have stored it
    user_name = req.session.get('user_name', 'Unknown')
    user_email = req.session.get('user_email', 'Unknown')
    user_contact = req.session.get('user_contact', 'Unknown')
    predicted_disease = req.session.get('predicted_disease', 'Unknown')
    model_accuracy = req.session.get('model_accuracy', '86.49')
    model_name = 'CNN'
    symptoms = req.session.get('symptoms', 'Unknown')
    medication = req.session.get('medication', 'Unknown')
    image_path = req.session.get('image_path', None)  # Assuming this is the path to the uploaded image

    # Create a response object with PDF content type
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="classification_result.pdf"'

    # Create a PDF document
    doc = SimpleDocTemplate(response, pagesize=A4)
    elements = []

    # Set up styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    normal_style = styles['Normal']

    # Title
    elements.append(Paragraph('Skin Cancer Detection With User Details', title_style))
    elements.append(Spacer(1, 12))

    # User details table
    user_data = [
        ['User Name', user_name],
        ['User Email', user_email],
        ['Predicted Disease', predicted_disease],
        ['Model Used', model_name],
        ['Model Accuracy', f"{model_accuracy}%"],
        ['Symptoms', symptoms],
        ['Medication', medication],
    ]

    # Create a Table object
    user_table = Table(user_data, colWidths=[150, 300])
    
    # Apply styling to the table
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(user_table)
    elements.append(Spacer(1, 12))

    # If an image is provided, add it to the PDF
    if image_path:
        elements.append(Paragraph('Predicted Image:', title_style))
        elements.append(Spacer(1, 12))

        # Add the image to the PDF
        img = Image(image_path, width=400, height=300)
        elements.append(img)
        elements.append(Spacer(1, 12))

    # Build the PDF
    doc.build(elements)
    
    return response



import re
import requests
from django.conf import settings
from django.shortcuts import render, redirect
from .models import Conversation
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def user_chatbot(request):
    conversations = Conversation.objects.all().order_by('created_at')
    
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if user_message:
            # Call Perplexity API
            headers = {
                "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers
            )
            
            bot_response = "Error: Could not get response from AI"
            if response.status_code == 200:
                try:
                    bot_response = response.json()['choices'][0]['message']['content']
                    
                    # Remove markdown bold () and any references (e.g., [1], [2], etc.)
                    bot_response = re.sub(r'\\([^]+)\\*', r'\1', bot_response)  # Remove bold
                    bot_response = re.sub(r'\[\d+\]', '', bot_response)  # Remove reference numbers
                except:
                    pass
                
            Conversation.objects.create(
                user_message=user_message,
                bot_response=bot_response
            )
            
            return redirect('chatbot')
    
    return render(request, 'user/chatbot.html', {'conversations': conversations})
