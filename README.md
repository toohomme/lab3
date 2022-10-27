# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Еремин Егор Константинович
- РИ000024
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | # | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)


## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1

Ход работы:
1.Добавляем ML Agent в проект:	

![image](https://user-images.githubusercontent.com/102966721/198173796-a44ad355-503e-490a-891e-1fcc8126a913.png)

2.Активируем ML-агента в Anaconda Promt:

![image](https://user-images.githubusercontent.com/102966721/198173840-dfb08e65-ef93-4b04-a7a6-858a4e107329.png)
![image](https://user-images.githubusercontent.com/102966721/198173855-500bdc8a-a6f6-479b-ab41-7a91f1f53b04.png)
![image](https://user-images.githubusercontent.com/102966721/198173862-a5f5d13b-626f-415e-9480-b293ab7258ad.png)

3. Создаем плоскость, куб и сферу:

![image](https://user-images.githubusercontent.com/102966721/198173887-2e2d61eb-e694-49ae-ada7-d8e87d3987a4.png)

4.Создаем скрипт и добавляем его к сфере:

![image](https://user-images.githubusercontent.com/102966721/198173928-e27e9ea6-dd80-4fe2-b184-384c7544e89b.png)

5. Добавляем в скрипт код:

``` C#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```

6.Объекту сфера добавляем компоненты Rigidbody, Decision Requester, Behavior Parameters и настраиваем их:

 ![image](https://user-images.githubusercontent.com/102966721/198174038-83cc0e9a-92d3-47f8-ae3c-2ceb5b70d761.png)

7. В корень проекта добавляем файл конфигурации нейронной сети:

![image](https://user-images.githubusercontent.com/102966721/198174069-3be6e199-f1e7-436b-bbe9-f373f720d6d8.png)

8. Запускаем работу ML-агента:

![image](https://user-images.githubusercontent.com/102966721/198174097-4b3772e4-d0f5-4882-952b-4b1fa222b2ad.png)

9. После этого запускаем проект на Unity и наблюдаем за процессом:

![image](https://user-images.githubusercontent.com/102966721/198174200-7e4b8c23-6093-4b82-8689-6b891968add3.png)

![image](https://user-images.githubusercontent.com/102966721/198174211-56847c00-62e4-4c61-b387-18efb16f669e.png)

10. Добавляем модель обучения:

![image](https://user-images.githubusercontent.com/102966721/198174266-c17f1f54-7618-4718-994e-adc68b120a37.png)

![image](https://user-images.githubusercontent.com/102966721/198174278-95dcca73-3951-4dc7-8030-9e515f51deb0.png)




## Задание 2
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

- Перечисленные в этом туториале действия могут быть выполнены запуском на исполнение скрипт-файла, доступного [в репозитории](https://github.com/Den1sovDm1triy/hfss-scripting/blob/main/ScreatingSphereInAEDT.py).
- Для запуска скрипт-файла откройте Ansys Electronics Desktop. Перейдите во вкладку [Automation] - [Run Script] - [Выберите файл с именем ScreatingSphereInAEDT.py из репозитория].

```py

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.Rename("C:/Users/denisov.dv/Documents/Ansoft/SphereDIffraction.aedt", True)
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateSphere(
	[
		"NAME:SphereParameters",
		"XCenter:="		, "0mm",
		"YCenter:="		, "0mm",
		"ZCenter:="		, "0mm",
		"Radius:="		, "1.0770329614269mm"
	], 
)

```

## Задание 3
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

- Перечисленные в этом туториале действия могут быть выполнены запуском на исполнение скрипт-файла, доступного [в репозитории](https://github.com/Den1sovDm1triy/hfss-scripting/blob/main/ScreatingSphereInAEDT.py).
- Для запуска скрипт-файла откройте Ansys Electronics Desktop. Перейдите во вкладку [Automation] - [Run Script] - [Выберите файл с именем ScreatingSphereInAEDT.py из репозитория].

```py

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.Rename("C:/Users/denisov.dv/Documents/Ansoft/SphereDIffraction.aedt", True)
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateSphere(
	[
		"NAME:SphereParameters",
		"XCenter:="		, "0mm",
		"YCenter:="		, "0mm",
		"ZCenter:="		, "0mm",
		"Radius:="		, "1.0770329614269mm"
	], 
)

```

## Выводы

Абзац умных слов о том, что было сделано и что было узнано.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
