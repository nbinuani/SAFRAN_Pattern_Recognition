# -*- coding utf-8 -*-
"""
MAJ : 25/06/18
S598658 Binuani Nicolas
"""
# % Librairies
from tkinter.filedialog import *
from tkinter import messagebox
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from src.submodules.lib_kmeans.k_mean import kmean_clustering
from src.submodules.lib_kmeans.k_mean_manual import kmean_clustering_manual
from tkinter import *

#######################################################################################################################
class Functions_tool():
    """
    Class with all the classifier & preparation data functions
    """
    def _loader(self):
        """
        Fonction qui permet d'ouvrir une fenetre et de chercher le fichier csv voulu
        :return: filepath = path+filename : le chemin et le nom du fichier
        """
        filepath = askopenfilename(title="Choisir le fichier csv", filetypes=[('csv files', '.csv'), ('all files', '.*')])
        return filepath

    def _clean_df(self, df):
        df.dropna(axis=0, inplace=True)
        return df

    def _liste_parameters(self, filepath):
        df = pd.read_csv(filepath, sep=',', header=0)
        self._clean_df(df)
        try:
            measure_type = pd.Series(data=df.measure_type)
        except AttributeError:
            print("Le format du CSV en entrée n'est pas bon ! Se référer au DOC.txt")
            sys.exit()
        else:
            measure_type = measure_type.drop_duplicates()
            list_param = []
            list_param = measure_type.values
            print(list_param)
            return list_param


    def _filter(self, df, liste_param, index):
        """
        Elimine les données profile du dataframe
        :param df: dataframe correspondant au fichier csv d'entrée
        :return: df : dataframe des flatness seulement
        """
        df.drop(df[df['measure_type'] != liste_param[index]].index, inplace=True)
        return df

    def _formatage(self, df, colonne):
        """
        Fonction qui permet de formater les données avec 50 colonnes de flatness
        (correspond aux x) et les serial en index et les données y dans le df
        :param df: dataframe à formater contenant serial/y/flatness(50 flatness qui se repetent)
                colonne : correpond aux 50 coupes (axes des abscisses)
        :return: le DF formaté
        """
        repet = int(len(df.y) / len(colonne))
        flatnessCR = []
        for i in range(0, repet):
            flatnessCR.extend(colonne)
        df['flatnessCR'] = flatnessCR
        df = df.pivot(index='serial', columns='flatnessCR', values='y')
        return df

    def _drop_column_nonuse(self, df):
        """
        fonction qui permet d'éliminer les colonnes non utilisées (measure_type, measure_dttm et x)
        :param df: dataframe d'entrée
        :return: dataframe avec seulement serial et y en colonne
        """
        df.drop(['measure_dttm', 'x'], axis=1, inplace=True)
        return df

    def _df_output(self, df_final, label):
        """
        fonction qui renvoie le dataframe de sortie avec Serial/Amplitude/label
        -création de 2 dataframe l'un avec les y_min l'autre les y_max (fonction abs(max) et abs(min))
        -on somme les 2 y min et max en une que l'on nomme Amplitude
        :param df_final: le dataframe formaté avec en values tous les y par serial (index)
                label : series contenant les labels selon les serial (index)
        :return: output : le dataframe selon Serial(index)/Amplitude/label
        """
        max_df = df_final.max(axis=1)
        max_df = abs(max_df)
        min_df = df_final.min(axis=1)
        min_df = abs(min_df)
        amp = max_df + min_df
        amplitude = pd.DataFrame(data=amp, columns=["amplitude"], index=df_final.index)
        output = pd.concat([amplitude, label], axis=1)
        return output

    def _visual(self, df, label, name_file, name_title):
        """
        Fonction qui permet d'enregistrer les HTML de chaque graphe (4)
        :param df: Dataframe intrados-top/bottom et extrados-top/bottom
        :param label: Series contenant les labels de chaque intrados-top/bottom et extrados-top/bottom
        :return: les HTML enregistrés dan sle même path que le csv lu
        """
        trace = []
        for k in range(0, len(df)):
            trace.append(
                go.Scatter(
                    x=df.columns.tolist(),
                    y=df.iloc[k, :],
                    mode="markers+lines",
                    marker=dict(color=label.iloc[k]),
                    name=df.index[k],
                    legendgroup=label.iloc[k]
                )
            )
        layout = go.Layout(title=name_title, xaxis=dict(title="Coupe CR"), yaxis=dict(title="y (mm)"))
        fig0 = go.Figure(data=trace, layout=layout)
        py.plot(fig0, filename=os.path.join(name_file), auto_open=False)

    def process_auto(self, DF, type, colorsbar, check):
        """
        Fonction qui permet d'éxecuter le process de clustering + la visualisation + les données output
        Ici la fonction le fait automatiquement (silhouette_score)
        :param DF: df du type de donnée que l'on veut clusteriser
        :param type:  nom du type de donnée étudiée
        :return: le fichier HTML (visual) + CSV (output)
        """
        df_nolabel = DF
        try:
            DF = kmean_clustering(DF, colorsbar)
        except ValueError:
            print('NO '+ type)
        else:
            label = pd.Series(data=DF.label, index=DF.index)
            self._visual(df=df_nolabel, label=label, name_file=type + '.html', name_title=type+' Pattern ')
            output = self._df_output(DF, label)
            if(check == True):
                output.to_csv('output_'+type+'.csv', sep=";", decimal=",")

            return DF, output

    def process_manual(self, DF, type, colorsbar, nb_cluster, check):
        """
         Fonction qui permet d'éxecuter le process de clustering + la visualisation + les données output
         Ici la fonction récupère le nombre de cluster voulu par le user
         :param DF: df du type de donnée que l'on veut clusteriser
         :param type:  nom du type d edonnée étudiée
         :return: le fichier HTML (visual) + CSV (output)
         """
        df_nolabel = DF
        try:
            DF = kmean_clustering_manual(DF, colorsbar, nb_cluster)
        except ValueError:
            print('NO ' + type)
        else:
            label = pd.Series(data=DF.label, index=DF.index)
            self._visual(df=df_nolabel, label=label, name_file=type + '.html', name_title=type+' Pattern')
            output = self._df_output(DF, label)
            if(check == True):
                output.to_csv('output_'+type+'.csv', sep=";", decimal=",")

            return DF, output

    def _process_sous_classe_auto(self, df, type, color_label, colorsbar, name_title, check_csv, output):
        """
        Fonction qui permet de clusteriser auto en nouvelles sous classes les classes précédentes
        :param df: dataframe avec les données de chaque famille qui a été clusterisé
        :param type: type de mesure qui servira à nommer le fichier en fonction de la mesure étudiée
        :param color_label: label correspondant a la famille dont la donnée a été clusterisé
        :param colorsbar: constante liste contenant les couleurs pour la clusterisation
        :param name_title: reprend le type de la donnée étudiée pour nommer le graphique
        :return:
        """
        cluster = pd.DataFrame(index=df.index, columns=df.columns, data=df.values)
        cluster.drop(cluster[cluster['label'] != color_label].index, inplace=True)
        cluster.drop('label', axis=1, inplace=True)
        try:
            cluster = kmean_clustering(cluster, colorsbar)
        except ValueError:
            print()
        else:
            label = pd.Series(data=cluster.label, index=cluster.index)
            cluster.drop('label', axis=1, inplace=True)
            self._visual(df=cluster, label=label, name_file=type + '_cluster_' + color_label + '.html',
                    name_title=type + name_title)
            if(check_csv == True):
                output_ssprocess = pd.concat([output, label], axis=1, sort=True)
                output_ssprocess.dropna(axis=0, inplace=True)
                output_ssprocess.to_csv('output_' + type + '_' + color_label + '.csv', sep=";", decimal=",")

    def _process_sous_classe_manual(self, df, type, color_label, colorsbar, name_title, nb_cluster, check_csv, output):
        """
        Functions ro clusterize each previous cluster
        :param df: dataframe of the previous cluster
        :param type: type de mesure qui servira à nommer le fichier en fonction de la mesure étudiée
        :param color_label: label correspondant a la famille dont la donnée a été clusterisé
        :param colorsbar: constante liste contenant les couleurs pour la clusterisation
        :param name_title: reprend le type de la donnée étudiée pour nommer le graphique
        :return:
        """
        cluster = pd.DataFrame(index=df.index, columns=df.columns, data=df.values)
        cluster.drop(cluster[cluster['label'] != color_label].index, inplace=True)
        cluster.drop('label', axis=1, inplace=True)
        try:
            cluster = kmean_clustering_manual(cluster, colorsbar, nb_cluster)
        except ValueError:
            print()
        else:
            label = pd.Series(data=cluster.label, index=cluster.index)
            cluster.drop('label', axis=1, inplace=True)
            self._visual(df=cluster, label=label, name_file=type + '_cluster_' + color_label + '.html',
                    name_title=type + name_title)
            if(check_csv == True):
                output_ssprocess = pd.concat([output, label], axis=1)
                output_ssprocess.dropna(axis=0, inplace=True)
                print(output_ssprocess)
                output_ssprocess.to_csv('output_' + type + '_' + color_label + '.csv', sep=";", decimal=",")

#######################################################################################################################
class Interface():
    """
    Class for the IHM & get the variables from
    """
    def _IHM(self):
        master = Tk()
        master.title("Configuration ")
        master.geometry("230x360")
        texte = IntVar()
        choix1 = Radiobutton(master, text='Auto ', variable=texte, value=0)
        choix2 = Radiobutton(master, text='Manuel ', variable=texte, value=1)
        choix1.pack()
        choix2.pack()

        checking = BooleanVar()
        check1 = Checkbutton(master, text=' Sous Cluster ', variable=checking)
        check1.pack()

        check_csv0 = BooleanVar()
        check0 = Checkbutton(master, text=' CSV du Premier Clustering ', variable=check_csv0)
        check0.pack()

        check_csv = BooleanVar()
        check2 = Checkbutton(master, text=' CSV du Second Clustering ', variable=check_csv)
        check2.pack()

        cluster1 = IntVar()
        scale1 = Scale(master, variable=cluster1, from_=2, to=10,
                      orient='horizontal', label='Nombre de cluster :',
                      tickinterval=1, width=20, troughcolor='white', sliderlength=20, length=200)
        scale1.pack()
        cluster2 = IntVar()
        scale2 = Scale(master, variable=cluster2, from_=2, to=10,
                      orient='horizontal', label='Nombre de sous cluster :',
                      tickinterval=1, width=20, troughcolor='white', sliderlength=20, length=200)
        scale2.pack()

        lb0 = Label(master, text='Nombre de CR par sn : ')
        lb0.pack()
        nb_CR = IntVar()
        nb_CR.set(50)
        entry0 = Entry(master, textvariable=nb_CR, width=20)
        entry0.pack()

        print(cluster1)

        button_quit = Button(text=" OK ", command=master.destroy).pack()
        master.protocol("WM_DELETE_WINDOW", self._exit)
        master.mainloop()
        return texte, cluster1, cluster2, checking, check_csv, check_csv0, nb_CR

    def _exit(self):
        if messagebox.askokcancel('Sortie', 'Souhaitez-vous tout stopper ?'):
            sys.exit()

    def get_variables(self):
        (choice_user, cluster_process, cluster_ssprocess, checking, check_csv, check_csv0, nb_CR) = self._IHM()
        choice = choice_user.get()
        nb_cluster1 = cluster_process.get()
        nb_cluster2 = cluster_ssprocess.get()
        check_sscluster = checking.get()
        ss_csv = check_csv.get()
        csv = check_csv0.get()
        coupes = nb_CR.get()
        return choice, nb_cluster1, nb_cluster2, check_sscluster, ss_csv, csv, coupes

#######################################################################################################################
class Controller():
    """
    Class to run all the tool
    """
    def run(self):
        func = Functions_tool()
        intf = Interface()
        choice, nb_cluster1, nb_cluster2, check, check_csv, check_csv0, coupes = intf.get_variables()
        ## We can add some colors so increase the limit of cluster's max number
        COLORS = ['red', 'green', 'blue', 'black', 'gray', 'magenta', 'turquoise', 'gold', 'sienna', 'olive']
        COLONNE = [i for i in range(0, coupes)]
        filepath = func._loader()
        liste_param = func._liste_parameters(filepath)
        df = {}
        DF = {}
        df_filter = {}
        DF_processing = {}
        first_output = {}
        for number_df in range(len(liste_param)):
            print(number_df)
            df[number_df] = pd.read_csv(filepath, sep=',', header=0)
            try:
                df[number_df] = func._drop_column_nonuse(df[number_df])
                df_filter[number_df] = func._filter(df[number_df], liste_param, number_df)
                df_filter[number_df].drop('measure_type', axis=1, inplace=True)
                DF[number_df] = func._formatage(df_filter[number_df], COLONNE)
                if (choice == 0):
                    (DF_processing[number_df], first_output[number_df]) = func.process_auto(DF[number_df], liste_param[number_df], colorsbar=COLORS, check=check_csv0)
                    if (check == True):
                        if (DF_processing[number_df].empty == False):
                            for m in COLORS:
                                func._process_sous_classe_auto(DF_processing[number_df], liste_param[number_df], m, colorsbar=COLORS, name_title=" Cluster " + m, check_csv=check_csv, output=first_output[number_df])

                if (choice == 1):
                    (DF_processing[number_df], first_output[number_df]) = func.process_manual(DF[number_df], liste_param[number_df], colorsbar=COLORS, nb_cluster=nb_cluster1, check=check_csv0)
                    if (check == True):
                        if (DF_processing[number_df].empty == False):
                            for m in COLORS:
                                func._process_sous_classe_manual(DF_processing[number_df], liste_param[number_df], m, colorsbar=COLORS, name_title=" Cluster " + m, nb_cluster=nb_cluster2, check_csv=check_csv, output=first_output[number_df])
            except:
                pass
        #     except:
        #         continue
        # print("Revérifier la configuration du fichier")

if __name__ == '__main__':
    c = Controller()
    c.run()