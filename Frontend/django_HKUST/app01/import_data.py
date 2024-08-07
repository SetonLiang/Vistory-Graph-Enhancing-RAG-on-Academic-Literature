import os
import csv
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_HKUST.settings')
django.setup()

from app01.models import PublicationDatasets


def import_data():
    print("Importing PublicationDatasets...")
    with open('datasets/CMA/zhangkang.csv', encoding='gbk', errors='replace') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            name = row[2]
            author = row[3].split(", ")
            for i in range(10):
                if i < len(author):
                    continue
                else:
                    author.append(None)
            source = row[1]
            years = row[0]
            doi = row[7]
            abstract = row[6]
            IdName = row[10]
            Citation = row[9]
            Keywords = row[11]
            PublicationDatasets.objects.create(Name=name, AuthorOne=author[0], AuthorTwo=author[1],
                                               AuthorThree=author[2], AuthorFour=author[3], AuthorFive=author[4],
                                               AuthorSix=author[5], AuthorSeven=author[6], AuthorEight=author[7],
                                               AuthorNine=author[8], AuthorTen=author[9], Sources=source,
                                               PublishedYears=years, Doi=doi, Abstracts=abstract, IdName=IdName,
                                               Citation=Citation, Keyword=Keywords)


def delete_data():
    print("Deleting PublicationDatasets...")
    PublicationDatasets.objects.all().delete()


if __name__ == '__main__':
    import_data()
    # delete_data()
