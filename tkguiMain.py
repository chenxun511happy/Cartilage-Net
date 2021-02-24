import tkgen.gengui
from tkinter import filedialog
try:
    from tkinter.constants import END
except ImportError:
    from Tkinter import END
from tkinter import messagebox
from DeeplearningMain import ProDeeplearningMain

def analysis_Map(checkValue = [], modelValue = 0):
    if checkValue[0] == 1:
        analysis_type = 0
    elif checkValue[1] == 1:
        analysis_type = 1
    elif checkValue[2] == 1:
        analysis_type = 2
    else:
        messagebox.showerror("error", "Segment Detection Classify check")
    analysis_type = {"model_type": modelValue,
                     "analysis_type": analysis_type,
                     "predict_type": checkValue[3],
                     "pixel_size": [checkValue[4], checkValue[5], checkValue[6]]
                     }

    return analysis_type


if __name__ == '__main__':
    root = tkgen.gengui.TkJson('tkgui.json', title='Gui-DeepLearning')

    def tkCNNw(event=None):
        print(v.get())
        print(c.get())
        print(m.get())
        if len(folder_config.get()) == 0 or len(folder_config1.get()) == 0:
            messagebox.showerror("error", "No folder Config")
        else:
            checkValue = [cv2.get(), cv3.get(), cv4.get(), c2.get(), v.get(), v1.get(), v2.get()]
            analysis_type = analysis_Map(checkValue, 0)
            ProDeeplearningMain(eval(folder_config.get()), eval(folder_config1.get()), analysis_type)
            root.destroy()
    def tkCNNu(event=None):
        print(v.get())
        print(c.get())
        print(m.get())
        if len(folder_config.get()) == 0 or len(folder_config1.get()) == 0:
            messagebox.showerror("error", "No folder Config")
        else:
            checkValue = [cv2.get(), cv3.get(), cv4.get(), c2.get(), v.get(), v1.get(), v2.get()]
            analysis_type = analysis_Map(checkValue, 1)
            ProDeeplearningMain(eval(folder_config.get()), eval(folder_config1.get()), analysis_type)
            root.destroy()
    def tkCNNwu(event=None):
        print(v.get())
        print(c.get())
        print(m.get())
        if len(folder_config.get()) == 0 or len(folder_config1.get()) == 0:
            messagebox.showerror("error", "No folder Config")
        else:
            checkValue = [cv2.get(), cv3.get(), cv4.get(), c2.get(), v.get(), v1.get(), v2.get()]
            analysis_type = analysis_Map(checkValue, 2)
            ProDeeplearningMain(eval(folder_config.get()), eval(folder_config1.get()), analysis_type)
            root.destroy()

    def chooseFile():
        root.sourceFolder = filedialog.askdirectory(parent=root, initialdir="/",
                                                         title='Please select a directory')
        p_entry = root.get('priv')
        p_entry.config(state='normal')
        p_entry.delete(0, END)
        p_entry.insert(END, repr(root.sourceFolder))
        p_entry.config(state='disabled')
        print(root.sourceFolder)

    def chooseFile1():
        root.sourceFolder = filedialog.askdirectory(parent=root, initialdir="/",
                                                    title='Please select a directory')
        pr_entry = root.get('pub')
        pr_entry.config(state='normal')
        pr_entry.delete(0, END)
        pr_entry.insert(END, repr(root.sourceFolder))
        pr_entry.config(state='disabled')
        print(root.sourceFolder)


    # config vars for checkboxes etc.
    c = root.checkbox('check', text=0)
    c1 = root.checkbox('check1', text=1)
    c2 = root.checkbox('check_test1', text=1)
    cv2 = root.checkbox('check_test2', text=0)
    cv3 = root.checkbox('check_test3', text=0)
    cv4 = root.checkbox('check_test4', text=1)
    c3 = root.checkbox('check_third', text=1)
    c4 = root.checkbox('check_third1', text=1)

    v = root.entry('entry', key='<Return>', cmd=tkCNNw, focus=True, text= '1024')
    v1 = root.entry('entry1', key='<Return>', cmd=tkCNNw, focus=True, text='1024')
    v2 = root.entry('entry2', key='<Return>', cmd=tkCNNw, focus=True, text='100')
    m = root.entry('entry_test', key='<Return>', cmd=tkCNNw, focus=True, text='chen')
    n = root.entry('entry_third', key='<Return>', cmd=tkCNNw, focus=True, text='chen')

    #folder config
    folder_config = root.entry('priv', key='<Return>', cmd=tkCNNw, focus=True,)
    folder_config1 = root.entry('pub', key='<Return>', cmd=tkCNNw, focus=True, )
    # import file
    root.button('chooseFolder', chooseFile)
    root.button('chooseFolder1', chooseFile1)

    # add button behaviour
    root.button('cnnwatershed', tkCNNw)
    root.button('cnnunet', tkCNNu)
    root.button('cnnunetwatershed', tkCNNwu)
    root.button('cancel', root.destroy)


    root.mainloop()
