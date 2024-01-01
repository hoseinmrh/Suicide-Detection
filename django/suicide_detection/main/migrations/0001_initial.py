# Generated by Django 3.2.23 on 2023-12-25 13:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SuicideTest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(verbose_name='text')),
                ('time', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='TestResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('suicidal_rate', models.IntegerField(verbose_name='Suicidal Rate')),
                ('non_suicidal_rate', models.IntegerField(verbose_name='Non Suicidal Rate')),
                ('test', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.suicidetest', verbose_name='Suicide Test')),
            ],
        ),
    ]
