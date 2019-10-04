from django import forms

# Definindo classe para selecionar idade
class SelecioneIdade(forms.Form):
    MY_CHOICES = (
        ('0', 'Adult'),
        ('1', 'Senior'),
        ('2', 'Adolescent'),
    )
    idade = forms.ChoiceField(
        label='Selecione a idade',
        choices=MY_CHOICES,
        widget=forms.Select(
            attrs={'class': 'styled-select yellow rounded'})) # Definindo as cores da interface gráfica

# Definindo classe para o grau de risco
class GrauRisco(forms.Form):
    MY_CHOICES = (
        ('0', 'Adventurous'),
        ('1', 'Cautious'),
        ('2', 'Psychopath'),
        ('3', 'Normal')
    )
    risco = forms.ChoiceField(
        label='Gosto de Risco',
        choices=MY_CHOICES,
        widget=forms.Select(
            attrs={'class': 'styled-select yellow rounded'}))

# Definindo classe para o modelo do veículo
class ModeloVeiculo(forms.Form):
    MY_CHOICES = (
        ('0', 'Economy'),
        ('1', 'FamilySedan'),
        ('2', 'SportsCar'),
        ('3', 'Luxury'),
        ('4', 'SuperLuxury')
    )
    veiculo = forms.ChoiceField(
        label='Modelo do veiculo',
        choices=MY_CHOICES,
        widget=forms.Select(
            attrs={'class': 'styled-select yellow rounded'}))
