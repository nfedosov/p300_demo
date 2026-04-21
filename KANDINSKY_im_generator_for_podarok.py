import asyncio
import nest_asyncio
import AsyncKandinsky as kandinsky

# Patch the running event loop
nest_asyncio.apply()

# Initialize the model
model = kandinsky.FusionBrainApi(kandinsky.ApiWeb("np_fedosov@list.ru", "2086schA1"))


list_of_dirs = ['candy','bear','book','guitar','skis']

list_of_prompts_rus = ["коробка шоколадных конфет в красивой упаковке","плюшевый медведь","раскрытая книга без излишеств","акустическая гитара лежит по диаганали", "коньки с лезвием"]


#коробка шоколадных конфет в красивой упаковке, в окрытой красной подарочной коробке с золотым бантом, на размытом фоне ёлка с гирляндами и новогодняя атмосфера

#categroies
# Define the async function
async def generate(prompt,dir_idx,order_idx):
    try:
        # "пшеничное поле бескрайнее"
        # "лес осень"
        # город пейзаж
        result = await model.text2image(prompt, style="DEFAULT", art_gpt=True)
        # Новый параметр art_gpt - это инструмент для автоматического улучшения промпта => улучшение качества картинки
    except ValueError as e:
        print(f"Error:\t{e}")
    else:
        # Save the generated image
        with open("elements/"+list_of_dirs[dir_idx]+"/"+str(order_idx)+".png", "wb") as f:
            f.write(result.getvalue())
        print("Done!")
        
    try:
        # "пшеничное поле бескрайнее"
        # "лес осень"
        # город пейзаж
        prompt_gift = prompt+', в раскрытой красной подарочной коробкой с золотым бантом, на размытом фоне ёлка с гирляндами и новогодняя атмосфера'
        if prompt == list_of_prompts_rus[3]:
            prompt_gift = prompt+', с повязанным красным бантом, на размытом фоне ёлка с гирляндами и новогодняя атмосфера'
            
        if prompt == list_of_prompts_rus[2]:
            prompt_gift = prompt+', с повязанным красным бантом, на размытом фоне ёлка с гирляндами и новогодняя атмосфера'
            
        
        result = await model.text2image(prompt_gift, style="DEFAULT", art_gpt=True)
        # Новый параметр art_gpt - это инструмент для автоматического улучшения промпта => улучшение качества картинки
    except ValueError as e:
        print(f"Error:\t{e}")
    else:
        # Save the generated image
        with open("elements/"+list_of_dirs[dir_idx]+"_gift"+"/"+str(order_idx)+".png", "wb") as f:
            f.write(result.getvalue())
        print("Done!")


start_idx = 13
N_pics = 50
# Run the async function
for i in range(start_idx,start_idx+N_pics):
    print(i)
    for j in range(len(list_of_prompts_rus)):
        asyncio.run(generate(list_of_prompts_rus[j],j,i))



