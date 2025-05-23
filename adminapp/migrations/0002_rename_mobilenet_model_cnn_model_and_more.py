# Generated by Django 4.2.7 on 2024-04-27 08:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('adminapp', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Mobilenet_model',
            new_name='Cnn_model',
        ),
        migrations.RenameModel(
            old_name='Densenet_model',
            new_name='Efficientnet_model',
        ),
        migrations.RenameModel(
            old_name='Vgg16_model',
            new_name='Inception_model',
        ),
        migrations.DeleteModel(
            name='Xception_model',
        ),
        migrations.RenameField(
            model_name='comparison_graph',
            old_name='MobileNet',
            new_name='Cnn',
        ),
        migrations.RenameField(
            model_name='comparison_graph',
            old_name='DenseNet',
            new_name='Efficientnet',
        ),
        migrations.RenameField(
            model_name='comparison_graph',
            old_name='Vgg16',
            new_name='Inception',
        ),
        migrations.RemoveField(
            model_name='comparison_graph',
            name='Xception',
        ),
        migrations.AlterModelTable(
            name='cnn_model',
            table='Cnn',
        ),
        migrations.AlterModelTable(
            name='efficientnet_model',
            table='Efficientnet',
        ),
        migrations.AlterModelTable(
            name='inception_model',
            table='Inception',
        ),
    ]
