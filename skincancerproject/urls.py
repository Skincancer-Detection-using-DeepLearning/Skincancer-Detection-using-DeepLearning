"""
URL configuration for wildfireproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mainapp import views as mainapp_views
from userapp import views as userapp_views
from adminapp import views as admin_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    #main
    path('admin/', admin.site.urls),
    path('',mainapp_views.index,name='index'),
    path('about-us',mainapp_views.about_us,name='about_us'),
    path('user-login',mainapp_views.user_login,name='user_login'),
    path('admin-login',mainapp_views.admin_login,name='admin_login'),
    path('contact-us',mainapp_views.contact_us,name='contact_us'),
    path('register',mainapp_views.register,name='register'),
    path('otp',mainapp_views.otp,name='otp'),
    path('insert-unet-data/', admin_views.insert_data),
    path('hospitals/', admin_views.hospitalslist, name='hospitalslist'),
    path('doctors/', admin_views.hospitalslist_doctors, name='hospitalslist_doctors'),
    #user
    path('user-dashboard',userapp_views.user_dashboard,name='user_dashboard'),
    path('user-profile',userapp_views.user_profile,name='user_profile'),
    path('Classification',userapp_views.Classification,name='Classification'),
    path('Classificationunet',userapp_views.Classification_unet,name='Classificationunet'),
    path('Classification-result',userapp_views.Classification_result,name='Classification_result'),
    path('user-feedback',userapp_views.user_feedback,name='user_feedback'),  
    path('user-logout',userapp_views.user_logout,name='user_logout'),
    path('user-user_chatbot',userapp_views.user_chatbot,name='chatbot'),
    path('download-pdf/', userapp_views.download_classification_data_pdf, name='download_classification_data_pdf'),
    #admin
    path('admin-dashboard',admin_views.admin_dashboard,name='admin_dashboard'),
    path('pending-users',admin_views.pending_users,name='pending_users'),
    path('all-users',admin_views.all_users,name='all_users'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('delete-user/<int:id>', admin_views.delete_user, name = 'delete_user'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),
    path('admin-feedback',admin_views.admin_feedback,name='admin_feedback'),
    path('sentiment-analysis', admin_views.sentiment_analysis, name = 'sentiment_analysis'),
    path('sentiment-analysis-graph',admin_views.sentiment_analysis_graph,name='sentiment_analysis_graph'),
    path('comparision-graph',admin_views.comparision_graph,name='comparision_graph'),
    path('efficientnet',admin_views.efficient,name='efficientnet'),
    path('efficient-btn',admin_views.efficient_btn,name='efficient_btn'),
    path('cnn',admin_views.cnn,name='cnn'),
    path('cnn-btn',admin_views.cnn_btn,name='cnn_btn'),
    path('Train-Test-Split',admin_views.Train_Test_Split,name='Train_Test_Split'),
    path('Train-Test-Split-Result',admin_views.Train_Test_Split_Result,name='Train_Test_Split_Result'),
    path('inception',admin_views.inception,name='inception'),
    path('inception-Result',admin_views.inception_btn,name='inception_Result'),
    path('unet',admin_views.unetplusplus,name='unet'),
    path('unetbnt-Result',admin_views.unetplusplus_bnt,name='unet_bnt'),
    
    
]+ static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
