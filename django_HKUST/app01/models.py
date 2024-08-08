from django.db import models


class PublicationDatasets(models.Model):
    Name = models.TextField()
    IdName = models.TextField()
    Citation = models.CharField(max_length=10)
    Keyword = models.TextField()
    AuthorOne = models.CharField(max_length=50, null=True)
    AuthorTwo = models.CharField(max_length=50, null=True)
    AuthorThree = models.CharField(max_length=50, null=True)
    AuthorFour = models.CharField(max_length=50, null=True)
    AuthorFive = models.CharField(max_length=50, null=True)
    AuthorSix = models.CharField(max_length=50, null=True)
    AuthorSeven = models.CharField(max_length=50, null=True)
    AuthorEight = models.CharField(max_length=50, null=True)
    AuthorNine = models.CharField(max_length=50, null=True)
    AuthorTen = models.CharField(max_length=50, null=True)
    Sources = models.TextField()
    PublishedYears = models.CharField(max_length=10)
    Doi = models.CharField(max_length=100)
    Abstracts = models.TextField()
