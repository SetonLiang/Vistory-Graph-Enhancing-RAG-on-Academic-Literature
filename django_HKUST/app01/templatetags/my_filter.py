import os

from django import template

register = template.Library()


@register.filter
def spilt_author_last(obj):
    # _, res = obj.split(' ', 1)
    print(obj)
    return obj


@register.filter
def spilt_author_first(obj):
    # res, _ = obj.split(' ', 1)
    # res = ',' + res
    print(obj)
    return 'res'


@register.filter
def length(obj):
    return len(obj)


@register.filter
def file_exists(file_path):
    full_path = os.path.join('app01/static/', file_path)
    return os.path.exists(full_path)
