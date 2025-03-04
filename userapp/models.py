from django.db import models
from mainapp.models import *
# Create your models here.
class Feedback(models.Model):
    Feed_id = models.AutoField(primary_key=True)
    Rating=models.CharField(max_length=100,null=True)
    Review=models.CharField(max_length=225,null=True)
    Sentiment=models.CharField(max_length=100,null=True)
    Reviewer=models.ForeignKey(UserModel, on_delete=models.CASCADE,null=True)
    datetime=models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'feedback_details'
        

class Skin_cancer_dataset(models.Model):
   Data_id = models.AutoField(primary_key=True)
   Image = models.ImageField(upload_to='media/') 
   class Meta:
        db_table = "Skin_cancer_dataset" 

class Conversation(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return f"User: {self.user_message[:50]}..."
    
    
class All_model(models.Model):
    model_id = models.AutoField(primary_key = True)
    model_Name=models.CharField(max_length = 10, null=True) 
    model_accuracy = models.CharField(max_length = 10)    
    class Meta:
        db_table = 'Allmodel'    