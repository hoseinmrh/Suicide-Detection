from django.db import models

class SuicideTest(models.Model):
    text = models.TextField(verbose_name='text')
    time = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.text
    
class TestResult(models.Model):
    test = models.ForeignKey(SuicideTest,on_delete = models.CASCADE,verbose_name='Suicide Test')
    suicidal_rate = models.IntegerField(verbose_name = 'Suicidal Rate')
    non_suicidal_rate = models.IntegerField(verbose_name = 'Non Suicidal Rate')

    