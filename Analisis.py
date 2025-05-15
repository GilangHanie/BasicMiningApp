import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tkinter as tk
from tkinter import ttk, filedialog
import csv

def on_enter(event):
        event.widget.config(background='#008000')  # Perubahan Warna (Sedang Hover)

def on_leave(event):
        event.widget.config(background='#006400')  # Perubahan Warna (Tidak Hover)   
        
class MiningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mining App")
        self.geometry("720x405")
        self.configure(background="#E0E0E0")

        self.selected_method = tk.StringVar()
        self.file_path = tk.StringVar()

        container = tk.Frame(self)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.pack(padx=10, pady=(0,10),  fill="x", expand=True)

        self.frames = {}

        for F in (StartPage, TabbedPage, RunPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def get_column_data(self, feature):
        if hasattr(self, 'loaded_dataframe'):
            return self.loaded_dataframe[feature].dropna().tolist()
        return []

    def get_unique_values(self, feature):
        if hasattr(self, 'loaded_dataframe'):
            return sorted(self.loaded_dataframe[feature].dropna().unique().tolist())
        return []

    def is_numeric(self, feature):
        if hasattr(self, 'loaded_dataframe'):
            return self.loaded_dataframe[feature].dropna().map(lambda x: isinstance(x, (int, float))).all()
        return False


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#E0E0E0")
        self.controller = controller
        ttk.Label(self, text="Mining App", background="#E0E0E0",
                  font=("Comic Sans MS", 50), foreground="#006400").pack(pady=(0, 30))
        ttk.Label(self, text="Pilih File Laporan", background="#E0E0E0",
                  font=("Arial", 14)).pack(padx=10, fill="x", expand=True)
        
        top_frame = tk.Frame(self, bg="#E0E0E0")
        top_frame.pack(fill="x", padx=10, pady=(5, 5))
        file_frame = tk.Frame(top_frame, bg="#E0E0E0")
        file_frame.pack(pady=(10, 10), fill="x", expand=True, padx=10)
        method_frame = tk.Frame(top_frame, bg="#E0E0E0")
        method_frame.pack(pady=(10, 10), fill="x", expand=True, padx=10)
        
        # File Path Variable
        self.file_path = controller.file_path
        self.file_entry = ttk.Entry(file_frame, cursor="arrow", textvariable=self.file_path, state='readonly')
        self.file_entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.file_entry.bind("<Button-1>", self.browse_file)

        # Folder icon – using emoji or an image
        icon = tk.Button(file_frame, text="📁", foreground="white", background="#006400", command=self.browse_file)
        icon.pack(side="right")
        icon.bind("<Enter>", on_enter)
        icon.bind("<Leave>", on_leave)

        methods = ['Association', 'Classification', 'Clusterization', 'Regression']
        for method in methods:
            # Tombol Browse File Explorer
            browse_button = tk.Button(method_frame, text=method, background="#006400", 
                foreground="white", font=("Tahoma", 14), command=lambda m=method: self.start_method(m))
            browse_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), ipady=10)
            browse_button.bind("<Enter>", on_enter)
            browse_button.bind("<Leave>", on_leave)

        load_button = tk.Button(top_frame, text="Load Model", background="#006400",
            foreground="white", font=("Tahoma", 12), command=self.load_model)
        load_button.pack(pady=(30, 0), ipadx=10, ipady=6)
        load_button.bind("<Enter>", on_enter)
        load_button.bind("<Leave>", on_leave)

    def browse_file(self, event=None):
        path = filedialog.askopenfilename()
        if path:
            self.controller.file_path.set(path) 

    def start_method(self, method):
        if self.controller.file_path.get():
            self.controller.selected_method.set(method)
            if method == "Association":
                self.controller.frames[TabbedPage].show_preparation(self.controller.file_path.get())
            elif method == "Clusterization":
                self.controller.frames[TabbedPage].show_preparation(self.controller.file_path.get())
            elif method == "Classification":
                self.controller.frames[TabbedPage].show_preparation(self.controller.file_path.get())
            elif method == "Regression":
                self.controller.frames[TabbedPage].show_preparation(self.controller.file_path.get())
            self.controller.show_frame(TabbedPage)

    def run_assoc_preprocess(self, selected_columns, file_path):
        try:
            # Load file again using the provided path
            df = pd.read_csv(file_path)
            receipt_col = selected_columns["transaction_id"]
            item_col = selected_columns["item_name"]
            qty_col = selected_columns["quantity"]

            # Clean and process
            df[item_col] = df[item_col].astype(str).str.strip()
            df.dropna(subset=[receipt_col], inplace=True)
            df[receipt_col] = df[receipt_col].astype(str)
            df = df[~df[receipt_col].str.contains('C')]

            # Group and pivot
            basket = (df.groupby([receipt_col, item_col])[qty_col] 
                .sum().unstack().reset_index().fillna(0).set_index(receipt_col))

            def encode_units(x):
                x = float(x)
                return 0 if x <= 0 else 1

            basket_sets = basket.map(encode_units)

            item_counts = basket_sets.sum(axis=0)
            df_table = pd.DataFrame({'Item': item_counts.index, 'Total Sold': item_counts.values})
            df_table = df_table.sort_values("Total Sold", ascending=False)

            self.controller.basket_sets = basket_sets
            self.controller.overview_data = df_table
            self.controller.file_name = file_path.split("/")[-1]
            self.controller.total_transactions = df[receipt_col].nunique()

            self.controller.show_frame(TabbedPage)
            self.controller.frames[TabbedPage].show_overview()

        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing Failed: {e}")

    def run_class_preprocess(self, selected_columns, file_path):
        try:
            target_col = selected_columns["target"]
            feature_cols = selected_columns["features"]
            df = pd.read_csv(file_path)
            self.controller.selected_columns = selected_columns
            df = df.dropna(subset=[target_col] + feature_cols)

            self.controller.encoders = {}
            for col in [target_col] + feature_cols:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.controller.encoders[col] = le

            for col in [target_col] + feature_cols:
                if df[col].dtype == 'object':
                    df[col] = le.fit_transform(df[col])

            # Split into X (features) and y (target)
            X = df[feature_cols]
            y = df[target_col]
            class_data = {
                "X": X,
                "y": y,
                "columns": {
                    "target": target_col,
                    "features": feature_cols
                },
                "original_df": df,
                "encoder" : self.controller.encoders
            }

            self.controller.file_name = file_path.split("/")[-1]
            self.controller.show_frame(TabbedPage)
            self.controller.frames[TabbedPage].show_overview()
            self.controller.class_data = class_data
        
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing Failed: {e}")

    def run_clust_preprocess(self, selected_columns, file_path):
        try:
            df = pd.read_csv(file_path)
            data = df[selected_columns["clusters"]].copy()
            self.controller.selected_columns = selected_columns
            for col in data.select_dtypes(include=["object", "category"]).columns:
                data[col] = LabelEncoder().fit_transform(data[col])

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            self.controller.clust_data = {
                "file_path": file_path,
                "clusters": selected_columns,
                "original_data": df,
                "processed_data": scaled_data
            }

            self.controller.preview = df[selected_columns["clusters"]].head(50)  # Preview top 50 rows
            self.controller.file_name = file_path.split("/")[-1]
            self.controller.show_frame(TabbedPage)
            self.controller.frames[TabbedPage].show_overview()

        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {e}")

    def run_regres_preprocess(self, selected_columns, file_path):
        try:
            target_col = selected_columns["target"]
            feature_cols = selected_columns["features"]
            df = pd.read_csv(file_path)
            self.controller.selected_columns = selected_columns
            df = df.dropna(subset=[target_col] + feature_cols)

            self.controller.encoders = {}
            for col in [target_col] + feature_cols:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.controller.encoders[col] = le

            # Split into X (features) and y (target)
            X = df[feature_cols]
            y = df[target_col]
            regress_data = {
                "X": X,
                "y": y,
                "columns": {
                    "target": target_col,
                    "features": feature_cols
                },
                "original_df": df,
                "encoder" : self.controller.encoders
            }

            self.controller.file_name = file_path.split("/")[-1]
            self.controller.show_frame(TabbedPage)
            self.controller.frames[TabbedPage].show_overview()
            self.controller.regress_data = regress_data
        
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing Failed: {e}")

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            with open(file_path, "rb") as f:
                model_package = joblib.load(f)

            # Assign loaded data to controller
            self.controller.loaded_model = model_package["model"]
            self.controller.loaded_features = model_package["features"]
            self.controller.loaded_target = model_package["target"]
            self.controller.selected_method.set(model_package["type"])
            self.controller.loaded_dtypes = model_package.get("dtypes", {})
            self.controller.loaded_unique_values = model_package.get("unique_values", {})
            self.controller.loaded_dataframe = model_package.get("dataframe", pd.DataFrame())
            self.controller.loaded_encoders = model_package.get("encoders", {})
            self.controller.show_frame(RunPage)
            self.controller.frames[RunPage].load_fields()
            
class TabbedPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#E0E0E0")
        self.controller = controller

        ttk.Label(self, textvariable=controller.selected_method,
                  font=("Arial", 18), background="#E0E0E0", foreground="#333").pack(pady=10)

        notebook = ttk.Notebook(self)
        notebook.pack(fill='x', expand=True, padx=10, pady=10)

        tabs = ["Preparation", "Overview", "Results", "Visualization"]
        self.tabs = {}
        for name in tabs:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=name)
            self.tabs[name] = frame
            frame.pack(fill='both', expand=True)  
            notebook.add(frame, text=name)
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            ttk.Label(frame, text=f"Content for {name}").grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def show_preparation(self, file_path):
        prepare = self.tabs["Preparation"]
        for widget in prepare.winfo_children():
            widget.destroy()

        # Read and show sample data
        df = pd.read_csv(file_path)
        self.controller.df = df
        sample_data = df.head()
        tree = ttk.Treeview(prepare, columns=list(sample_data.columns), show="headings", height=6)

        for col in sample_data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=40, anchor='center')

        for _, row in sample_data.iterrows():
            tree.insert("", "end", values=list(row))

        tree.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 20), sticky="ew")

        column_names = df.columns.tolist()

        if self.controller.selected_method.get() == "Association":
            label1 = tk.Label(prepare, text="Select Transaction ID Column:", anchor='w')
            label1.grid(row=1, column=0, sticky='e', padx=(20, 10))
            dropdown1 = ttk.Combobox(prepare, values=column_names)
            dropdown1.grid(row=1, column=1, sticky='w', padx=10)

            label2 = tk.Label(prepare, text="Select Item Name Column:", anchor='w')
            label2.grid(row=2, column=0, sticky='e', padx=10)
            dropdown2 = ttk.Combobox(prepare, values=column_names)
            dropdown2.grid(row=2, column=1, sticky='w', padx=10)

            label3 = tk.Label(prepare, text="Select Quantity Column:", anchor='w')
            label3.grid(row=3, column=0, sticky='e', padx=10)
            dropdown3 = ttk.Combobox(prepare, values=column_names)
            dropdown3.grid(row=3, column=1, sticky='w', padx=10)
        
        elif self.controller.selected_method.get() == "Classification":
            label1 = tk.Label(prepare, text="Select Target Column:", anchor='w')
            label1.grid(row=1, column=0, sticky='e', padx=(20, 10))
            target_dropdown = ttk.Combobox(prepare, values=column_names)
            target_dropdown.grid(row=1, column=1, sticky='w', padx=10)

            feature_dropdowns = []
            for i in range(5):
                label = tk.Label(prepare, text=f"Select Feature Column {i+1}:", anchor='w')
                label.grid(row=2 + i, column=0, sticky='e', padx=10)
                dropdown = ttk.Combobox(prepare, values=column_names)
                dropdown.grid(row=2 + i, column=1, sticky='w', padx=10)
                feature_dropdowns.append(dropdown)

        elif self.controller.selected_method.get() == "Clusterization":
            cluster_dropdowns = []
            for i in range(5):
                label = tk.Label(prepare, text=f"Select Cluster Column {i+1}:", anchor='w')
                label.grid(row=1 + i, column=0, sticky='e', padx=10)
                dropdown = ttk.Combobox(prepare, values=column_names)
                dropdown.grid(row=1 + i, column=1, sticky='w', padx=10)
                cluster_dropdowns.append(dropdown)
        
        elif self.controller.selected_method.get() == "Regression":
            label1 = tk.Label(prepare, text="Select Target Column:", anchor='w')
            label1.grid(row=1, column=0, sticky='e', padx=(20, 10))
            target_dropdown = ttk.Combobox(prepare, values=column_names)
            target_dropdown.grid(row=1, column=1, sticky='w', padx=10)

            feature_dropdowns = []
            for i in range(5):
                label = tk.Label(prepare, text=f"Select Feature Column {i+1}:", anchor='w')
                label.grid(row=2 + i, column=0, sticky='e', padx=10)
                dropdown = ttk.Combobox(prepare, values=column_names)
                dropdown.grid(row=2 + i, column=1, sticky='w', padx=10)
                feature_dropdowns.append(dropdown)
        
        proceed_button = tk.Button(prepare, text="Set Column", background="#006400", foreground="white",
                                font=("Tahoma", 12), command=lambda: proceed_to_overview())
        proceed_button.grid(row=7, column=0, columnspan=2, pady=20)
        proceed_button.bind("<Enter>", on_enter)
        proceed_button.bind("<Leave>", on_leave)

        # Grid expansion behavior
        prepare.grid_columnconfigure(0, weight=1)
        prepare.grid_columnconfigure(1, weight=1)
        prepare.grid_rowconfigure(0, weight=0)

        # Proceed logic
        def proceed_to_overview():
            if self.controller.selected_method.get() == "Association":
                selected_columns = {
                    "transaction_id": dropdown1.get(),
                    "item_name": dropdown2.get(),
                    "quantity": dropdown3.get(),
                }
                self.controller.frames[StartPage].run_assoc_preprocess(selected_columns, file_path)

            elif self.controller.selected_method.get() == "Classification":
                selected_columns = {
                    "target": target_dropdown.get(),
                    "features": [d.get() for d in feature_dropdowns if d.get()]
                }
                self.controller.frames[StartPage].run_class_preprocess(selected_columns, file_path)

            elif self.controller.selected_method.get() == "Clusterization":
                selected = [d.get() for d in cluster_dropdowns if d.get()]
                selected_columns = {
                    "clusters": selected
                }
                self.controller.frames[StartPage].run_clust_preprocess(selected_columns, file_path)

            elif self.controller.selected_method.get() == "Regression":
                selected_columns = {
                    "target": target_dropdown.get(),
                    "features": [d.get() for d in feature_dropdowns if d.get()]
                }
                self.controller.frames[StartPage].run_regres_preprocess(selected_columns, file_path)

    def show_overview(self):
        overview = self.tabs["Overview"]
        for widget in overview.winfo_children():
            widget.destroy()
            
        if self.controller.selected_method.get() == "Association":
            transactions = self.controller.basket_sets.apply(lambda row: row[row > 0].index.tolist(), axis=1).tolist()
            all_items = [item for sublist in transactions for item in sublist if pd.notnull(item)]
            item_counts = pd.Series(all_items).value_counts()
            self.min_support = tk.DoubleVar(value=0.01)
            self.min_confidence = tk.DoubleVar(value=0.5)

            top_frame = ttk.Frame(overview)
            top_frame.pack(fill="x", padx=10, pady=(5, 5))

            left_frame = ttk.Frame(top_frame)
            left_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
            middle_frame = ttk.Frame(top_frame)
            middle_frame.pack(side="left", fill="x", expand=True, padx=(10, 10))
            right_frame = ttk.Frame(top_frame)
            right_frame.pack(side="left", anchor="n", padx=(0, 10))

            # File Info
            ttk.Label(left_frame, text=f"File Name: {self.controller.file_name}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Total Transactions: {len(transactions)}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Unique Items: {item_counts.count()}", font=("Arial", 12)).pack(anchor="w")
            self.apriori_status_label = ttk.Label(left_frame, text="Apriori Completed", font=("Arial", 10), foreground="green")
            self.apriori_status_label.pack(anchor="w")
            self.apriori_status_label.pack_forget()  # Hide initially

            ttk.Label(middle_frame, text="Min Support").pack(anchor="w")
            support_entry = ttk.Entry(middle_frame, width=10, textvariable=self.min_support)
            support_entry.pack(anchor="w", pady=(0, 10))
            ttk.Label(middle_frame, text="Min Confidence").pack(anchor="w")
            confidence_entry = ttk.Entry(middle_frame, width=10, textvariable=self.min_confidence)
            confidence_entry.pack(anchor="w", pady=(0, 10))
            run_button = tk.Button(middle_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Run Apriori", command=lambda: self.run_apriori(self.min_support.get(), self.min_confidence.get()))
            run_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            export_button = tk.Button(right_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Export", command=self.export_results)
            export_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            run_button.bind("<Enter>", on_enter)
            run_button.bind("<Leave>", on_leave)
            export_button.bind("<Enter>", on_enter)
            export_button.bind("<Leave>", on_leave)

            # Table
            ttk.Label(overview, text="Top Items Sold", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
            table_frame = ttk.Frame(overview)
            table_frame.pack(fill="both", expand=True, padx=10, pady=5)
            columns = ("Item", "Total Sold")
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
            tree.heading("Item", text="Item")
            tree.heading("Total Sold", text="Total Sold")

            # Scrollbar
            v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=v_scrollbar.set)
            tree.pack(side="left", fill="both", expand=True)
            v_scrollbar.pack(side="right", fill="y")

            # Fill Table
            for item, count in item_counts.head(50).items():
                tree.insert("", "end", values=(item, count))

        elif self.controller.selected_method.get() == "Classification":
            self.train_data = tk.DoubleVar(value=80)
            top_frame = ttk.Frame(overview)
            top_frame.pack(fill="x", padx=10, pady=(5, 5))
            target_col = self.controller.selected_columns['target']
            total_classes = self.controller.df[target_col].nunique()

            left_frame = ttk.Frame(top_frame)
            left_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
            middle_frame = ttk.Frame(top_frame)
            middle_frame.pack(side="left", fill="x", expand=True, padx=(10, 10))
            right_frame = ttk.Frame(top_frame)
            right_frame.pack(side="left", anchor="n", padx=(0, 10))

            ttk.Label(left_frame, text=f"File Name: {self.controller.file_name}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Total Samples: {len(self.controller.df)}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Features Used: {', '.join(self.controller.selected_columns['features'])}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Target: {self.controller.selected_columns['target']}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Total Classes: {total_classes}", font=("Arial", 12)).pack(anchor="w")
        
            ttk.Label(middle_frame, text="Training Data Percentage").pack(anchor="w")
            train_entry = ttk.Entry(middle_frame, width=10, textvariable=self.train_data)
            train_entry.pack(anchor="w", pady=(0, 10))
            run_button = tk.Button(middle_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Run Decision Tree", command=lambda: self.run_dec_tree(self.train_data.get()))
            run_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")
            self.dec_tree_status_label = ttk.Label(middle_frame, text="Decision Tree Completed", font=("Arial", 10), foreground="green")
            self.dec_tree_status_label.pack(fill=tk.Y, anchor="w")
            self.dec_tree_status_label.pack_forget()  # Hide initially
        
            export_button = tk.Button(right_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Export", command=self.export_results)
            export_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")
            save_button = tk.Button(right_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Save Model", command=lambda: self.save_model(
                    self.controller.selected_columns['target'],
                    self.controller.selected_columns['features']
                ))
            save_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            run_button.bind("<Enter>", on_enter)
            run_button.bind("<Leave>", on_leave)
            export_button.bind("<Enter>", on_enter)
            export_button.bind("<Leave>", on_leave)
            save_button.bind("<Enter>", on_enter)
            save_button.bind("<Leave>", on_leave)

            target_series = self.controller.df[self.controller.selected_columns['target']]
            class_counts = target_series.value_counts().head(50)
            
            ttk.Label(overview, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
            table_frame = ttk.Frame(overview)
            table_frame.pack(fill="both", expand=True, padx=10, pady=5)
            columns = (f"{self.controller.selected_columns['target']} Class", "Total")
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
            tree.heading(f"{self.controller.selected_columns['target']} Class", text=f"{self.controller.selected_columns['target']} Class ")
            tree.heading("Total", text="Total")

            # Scrollbar
            v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=v_scroll.set)
            tree.pack(side="left", fill="x", expand=True)
            v_scroll.pack(side="right", fill="y")

            # Fill Table
            for cls, count in class_counts.items():
                tree.insert("", "end", values=(cls, count))

        elif self.controller.selected_method.get() == "Clusterization":
            self.total_cluster = tk.IntVar(value=3)
            top_frame = ttk.Frame(overview)
            top_frame.pack(fill="x", padx=10, pady=(5, 5))

            left_frame = ttk.Frame(top_frame)
            left_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
            middle_frame = ttk.Frame(top_frame)
            middle_frame.pack(side="left", fill="x", expand=True, padx=(10, 10))
            right_frame = ttk.Frame(top_frame)
            right_frame.pack(side="left", anchor="n", padx=(0, 10))

            ttk.Label(left_frame, text=f"File Name: {self.controller.file_name}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Total Samples: {len(self.controller.df)}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Clusters: {', '.join(self.controller.selected_columns['clusters'])}", font=("Arial", 12)).pack(anchor="w")
            self.k_means_status_label = ttk.Label(middle_frame, text="K-Means Completed", font=("Arial", 10), foreground="green")
            self.k_means_status_label.pack(fill=tk.Y, anchor="w")
            self.k_means_status_label.pack_forget()

            ttk.Label(middle_frame, text="Clusters Created").pack(anchor="w")
            cluster_entry = ttk.Entry(middle_frame, width=10, textvariable=self.total_cluster)
            cluster_entry.pack(anchor="w", pady=(0, 10))
            run_k_means_btn = tk.Button(middle_frame, text="Run K-Means", background="#006400", foreground="white",
                            font=("Tahoma", 10), command=lambda: self.run_k_means(self.total_cluster.get()))
            run_k_means_btn.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            export_button = tk.Button(right_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                                    text="Export", command=self.export_results)
            export_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            run_button.bind("<Enter>", on_enter)
            run_button.bind("<Leave>", on_leave)
            export_button.bind("<Enter>", on_enter)
            export_button.bind("<Leave>", on_leave)

            ttk.Label(overview, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
            tree = ttk.Treeview(overview, columns=list(self.controller.preview.columns), show="headings", height=20)
            tree.pack(side="left", fill="x", expand=True)
            for col in self.controller.preview.columns:
                tree.heading(col, text=col)
                tree.column(col, width=40, anchor='w')

            for _, row in self.controller.preview.iterrows():
                tree.insert("", "end", values=list(row))

        elif self.controller.selected_method.get() == "Regression":
            self.train_data = tk.DoubleVar(value=80)
            top_frame = ttk.Frame(overview)
            top_frame.pack(fill="x", padx=10, pady=(5, 5))
            target_col = self.controller.selected_columns['target']
            
            left_frame = ttk.Frame(top_frame)
            left_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))
            middle_frame = ttk.Frame(top_frame)
            middle_frame.pack(side="left", fill="x", expand=True, padx=(10, 10))
            right_frame = ttk.Frame(top_frame)
            right_frame.pack(side="left", anchor="n", padx=(0, 10))

            ttk.Label(left_frame, text=f"File Name: {self.controller.file_name}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Total Samples: {len(self.controller.df)}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Features Used: {', '.join(self.controller.selected_columns['features'])}", font=("Arial", 12)).pack(anchor="w")
            ttk.Label(left_frame, text=f"Target: {self.controller.selected_columns['target']}", font=("Arial", 12)).pack(anchor="w")
            
            ttk.Label(middle_frame, text="Training Data Percentage").pack(anchor="w")
            train_entry = ttk.Entry(middle_frame, width=10, textvariable=self.train_data)
            train_entry.pack(anchor="w", pady=(0, 10))
            run_button = tk.Button(middle_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Run Linear Regression", command=lambda: self.run_linear_regress(self.train_data.get()))
            run_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            self.linear_regres_status_label = ttk.Label(middle_frame, text="Linear Regression Completed", font=("Arial", 10), foreground="green")
            self.linear_regres_status_label.pack(fill=tk.Y, anchor="w")
            self.linear_regres_status_label.pack_forget()  # Hide initially
        
            export_button = tk.Button(right_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Export", command=self.export_results)
            export_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")
            save_button = tk.Button(right_frame, background="#006400", foreground="white", font=("Tahoma", 10),
                text="Save Model", command=lambda: self.save_model(
                    self.controller.selected_columns['target'],
                    self.controller.selected_columns['features']
                ))
            save_button.pack(fill=tk.Y, expand=True, padx=(0, 5), ipady=3, anchor="w")

            run_button.bind("<Enter>", on_enter)
            run_button.bind("<Leave>", on_leave)
            export_button.bind("<Enter>", on_enter)
            export_button.bind("<Leave>", on_leave)
            save_button.bind("<Enter>", on_enter)
            save_button.bind("<Leave>", on_leave)

            # Class distribution
            target_series = self.controller.df[self.controller.selected_columns['target']]
            class_counts = target_series.value_counts().head(50)
            
            ttk.Label(overview, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
            table_frame = ttk.Frame(overview)
            table_frame.pack(fill="both", expand=True, padx=10, pady=5)
            columns = (f"{self.controller.selected_columns['target']} Class", "Total")
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
            tree.heading(f"{self.controller.selected_columns['target']} Class", text=f"{self.controller.selected_columns['target']} Class ")
            tree.heading("Total", text="Total")

            # Scrollbar
            v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=v_scroll.set)
            tree.pack(side="left", fill="x", expand=True)
            v_scroll.pack(side="right", fill="y")

            # Fill Table
            for cls, count in class_counts.items():
                tree.insert("", "end", values=(cls, count))

    def run_apriori(self, support, confidence):
        try:
            basket_sets = self.controller.basket_sets
            support = float(support)
            confidence = float(confidence)

            frequent_itemsets = apriori(basket_sets, min_support=support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
            rules_filtered = rules.drop(['conviction', 'leverage', 'zhangs_metric',
                   'representativity', 'jaccard', 'kulczynski', 'certainty'], axis=1)
            rules_filtered['antecedents'] = rules_filtered['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_filtered['consequents'] = rules_filtered['consequents'].apply(lambda x: ', '.join(list(x)))
            self.controller.rules_filtered = rules_filtered

            self.show_results()
            self.show_visualization()
            self.apriori_status_label.pack(anchor="w", pady=(5, 0))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Apriori: {e}")

    def run_dec_tree(self, train_percent):
        data = self.controller.class_data
        X = data["X"]
        y = data["y"]

        # Split data
        test_size = 1 - (train_percent / 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        self.controller.class_model = clf

        # Store results (you can use these in the Results tab later)
        self.controller.class_data["results"] = {
            "model": clf,
            "accuracy": acc,
            "report": report,
            "conf_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred
        }

        # Show status message
        self.show_results()
        self.show_visualization()
        self.dec_tree_status_label.pack(anchor="w", pady=(5, 0))

    def run_k_means(self, total_cluster):
        try:
            df = self.controller.clust_data["original_data"].copy()
            scaled_data = self.controller.clust_data["processed_data"]
            selected_columns = self.controller.selected_columns["clusters"]
            
            # Prioritize numeric columns within the run_k_means method
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_columns) < 2:
                messagebox.showinfo("Info", "Please select at least two numeric columns for clustering.")
                return  # Exit if not enough numeric columns
            
            valid_columns = [col for col in selected_columns if col in numeric_columns]
            if len(valid_columns) < 2:
                messagebox.showinfo("Info", "At least two numeric columns are required. Proceeding with available columns.")
                valid_columns = numeric_columns[:2]
            if len(valid_columns) < 2:
                return
            
            self.controller.x_col, self.controller.y_col = valid_columns[0], valid_columns[1]
            kmeans = KMeans(n_clusters=total_cluster, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)

            df["Cluster"] = cluster_labels
            self.controller.clust_data["kmeans_model"] = kmeans
            self.controller.clust_data["result_df"] = df
            self.controller.df = df  # For visualization

            cluster_summary = df["Cluster"].value_counts().reset_index()
            cluster_summary.columns = ["Cluster", "Count"]
            self.controller.cluster_summary = cluster_summary

            self.show_results()
            self.show_visualization()
            self.k_means_status_label.pack(anchor="w", pady=(5, 0))

        except Exception as e:
            messagebox.showerror("Error", f"KMeans failed: {e}")

    def run_linear_regress(self, train_percent):
        try:
            data = self.controller.regress_data
            X = data["X"]
            y = data["y"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1 - train_percent / 100), random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results in controller
            result_df = X_test.copy()
            result_df["Actual"] = y_test.values
            result_df["Predicted"] = y_pred
            self.controller.regress_results = result_df
            self.controller.regress_model = model
            self.controller.regress_metrics = {
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }

            self.linear_regres_status_label.pack(anchor="w", pady=(5, 0))  # Show completion label
            self.show_results()
            self.show_visualization()

        except Exception as e:
            messagebox.showerror("Error", f"Linear Regression Failed: {e}")

    def show_results(self):
        result = self.tabs["Results"]
        for widget in result.winfo_children():
            widget.destroy()
        
        if self.controller.selected_method.get() == "Association":
            ttk.Label(result, text="Association Rules", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5)) 
            table_frame = ttk.Frame(result)
            table_frame.pack(fill="both", expand=True, padx=10, pady=5)
            columns = ("Antecedents", "Consequents", "Support", "Confidence", "Ante. Support", "Cons. Support", "Lift")
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)

            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="w")
            
            rules_sorted = self.controller.rules_filtered.sort_values(by="support", ascending=False)
            for col in ["support", "confidence", "antecedent support", "consequent support", "lift"]:
                rules_sorted[col] = rules_sorted[col].round(6)

            v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=v_scrollbar.set)
            tree.pack(side="left", fill="both", expand=True)
            v_scrollbar.pack(side="right", fill="y")

            for _, row in rules_sorted.iterrows():
                tree.insert("", "end", values=(
                    row["antecedents"],
                    row["consequents"],
                    row["support"],
                    row["confidence"],
                    row["antecedent support"],
                    row["consequent support"],
                    row["lift"]
                ))

        elif self.controller.selected_method.get() == "Classification":
            classified = self.controller.class_data.get("results", {})
            if not classified:
                ttk.Label(result, text="No results to display yet.", font=("Arial", 12)).pack(pady=10)
                return

            ttk.Label(result, text=f"Accuracy: {classified['accuracy']:.2f}", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

            report_df = pd.DataFrame(classification_report(
                classified["y_test"],
                classified["y_pred"],
                output_dict=True,
                zero_division=0
            )).transpose().round(2)
            report_df = report_df.loc[~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])]
            
            # Treeview table
            tree_frame = ttk.Frame(result)
            tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

            columns = list(report_df.columns)
            tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100, anchor="center")

            for idx, row in report_df.iterrows():
                tree.insert("", "end", values=list(row))  # Insert the data rows directly

            # Add vertical scrollbar
            vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")

            tree.pack(side="left", fill="both", expand=True)

        elif self.controller.selected_method.get() == "Clusterization":
            columns = list(self.controller.cluster_summary.columns)
            tree = ttk.Treeview(result, columns=columns, show="headings")
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="w", width=100)

            for cluster in self.controller.cluster_summary["Cluster"]:
                cluster_data = self.controller.df[self.controller.df["Cluster"] == cluster]
                count = cluster_data.shape[0]
                tree.insert("", "end", values=[cluster, count])

            tree.pack(fill="both", expand=True, padx=10, pady=10)

        elif self.controller.selected_method.get() == "Regression":
            columns = list(self.controller.regress_results.columns)
            tree = ttk.Treeview(result, columns=columns, show="headings")
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=100)

            for _, row in self.controller.regress_results.iterrows():
                tree.insert("", "end", values=list(row))

            tree.pack(fill="both", expand=True, padx=10, pady=10)

    def show_visualization(self):
        visual = self.tabs["Visualization"]
        for widget in visual.winfo_children():
            widget.destroy()

        if self.controller.selected_method.get() == "Association":
            self.figure = plt.Figure(figsize=(7, 5))
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, visual)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.ax.clear()
            if self.controller.rules_filtered.empty:
                return

            total_transactions = getattr(self.controller, 'total_transactions', 0)
            if total_transactions == 0:
                return

            # Get the total sold values for the items in the antecedents and consequents
            item_counts = self.controller.overview_data.set_index('Item')['Total Sold'].to_dict()
            top_rules = self.controller.rules_filtered.nlargest(10, 'support')
            x = range(len(top_rules))
            labels = [f"{a}→{c}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])]

            # Calculate the total sold for each rule combo
            total_sold = [
                (rule_support * total_transactions)
                for rule_support in top_rules['support']
            ]
            self.ax.bar(x, total_sold, color="#6495ED")
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            self.ax.set_title("Top 10 Association Rules by Total Sold")
            self.ax.set_ylabel("Total Sold")

            self.figure.tight_layout()
            self.canvas.draw()

        elif self.controller.selected_method.get() == "Classification":
            classified = self.controller.class_data.get("results", {})
            if not classified:
                ttk.Label(visual, text="No results to display yet.", font=("Arial", 12)).pack(pady=10)
                return

            cm = classified["conf_matrix"]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")

            canvas = FigureCanvasTkAgg(fig, master=visual)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        elif self.controller.selected_method.get() == "Clusterization":
            df = self.controller.df
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_columns) < 2:
                messagebox.showinfo("Info", "Not enough numeric columns to plot. Clustering results might be limited.")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_clusters = sorted(df["Cluster"].unique())
            colors = plt.cm.get_cmap("tab10", len(unique_clusters))
            for idx, cluster in enumerate(unique_clusters):
                cluster_data = df[df["Cluster"] == cluster]
                label = f"Cluster {cluster}" if cluster != -1 else "Noise"
                color = 'yellow' if cluster == -1 else colors(idx)
                ax.scatter(
                    cluster_data[self.controller.x_col], cluster_data[self.controller.y_col],
                    label=label, c=[color], edgecolor='k', s=50
                )

            ax.set_xlabel(self.controller.x_col)
            ax.set_ylabel(self.controller.y_col)
            ax.set_title("Clustering Result")
            ax.legend(loc="best")

            canvas = FigureCanvasTkAgg(fig, master=visual)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        elif self.controller.selected_method.get() == "Regression":
            results = self.controller.regress_results

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results["Actual"], results["Predicted"], alpha=0.6)
            ax.plot([results["Actual"].min(), results["Actual"].max()],
                    [results["Actual"].min(), results["Actual"].max()],
                    'r--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"R²: {self.controller.regress_metrics['R2']:.2f} | RMSE: {math.sqrt(self.controller.regress_metrics['MSE']):.2f}")

            canvas = FigureCanvasTkAgg(fig, master=visual)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_results(self):
        if self.controller.selected_method.get() == "Association":
            if self.controller.rules_filtered.empty:
                messagebox.showerror("Error", "No results to export. Run Apriori first.")
                return

            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return 

            self.controller.rules_filtered.to_csv(file_path, index=False)

        elif self.controller.selected_method.get() == "Classification":
            classified = self.controller.class_data.get("results", {})
            if not classified:
                messagebox.showerror("Error", "No results to export. Run classification first.")
                return

            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return

            report_dict = classified["report"]
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.round(2)
            report_df.loc["overall_accuracy"] = ["", "", "", classified["accuracy"]]
            report_df.to_csv(file_path)

        elif self.controller.selected_method.get() == "Clusterization":
            cluster_summary = self.controller.cluster_summary
            cluster_data = []

            for cluster in cluster_summary["Cluster"]:
                cluster_rows = self.controller.df[self.controller.df["Cluster"] == cluster]
                count = cluster_rows.shape[0]
                cluster_data.append([cluster, count])

            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return
            
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Cluster", "Count"])
                writer.writerows(cluster_data)

        elif self.controller.selected_method.get() == "Regression":
            if self.controller.regress_results.empty:
                messagebox.showerror("Error", "No results to export. Run Linear Regression first.")
                return
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return 

            self.controller.regress_results.to_csv(file_path, index=False)

    def save_model(self, target, features):
        try:
            if self.controller.selected_method.get() == "Regression":
                model = self.controller.regress_model
            elif self.controller.selected_method.get() == "Classification":
                model = self.controller.class_model
                
            file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])      
            if file_path:
                unique_values = {}
                dtypes = {}
                for col in features:
                    dtypes[col] = str(self.controller.df[col].dtype)
                    if self.controller.df[col].dtype == 'object' or not np.issubdtype(self.controller.df[col].dtype, np.number):
                        unique_values[col] = self.controller.df[col].dropna().unique().tolist()

                model_package = {
                    "model": model,
                    "features": features,
                    "target": target,
                    "type": self.controller.selected_method.get(),
                    "dtypes": dtypes,
                    "unique_values": unique_values,
                    "dataframe": self.controller.df,
                    "encoders": self.controller.encoders
                }
                joblib.dump(model_package, file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")

class RunPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#F0F0F0")
        self.controller = controller

        self.title_label = ttk.Label(self, font=("Arial", 16, "bold"), background="#F0F0F0")
        self.title_label.pack(pady=20)
        self.input_frame = tk.Frame(self, bg="#F0F0F0")
        self.input_frame.pack(padx=20, pady=10)

        self.entry_vars = []
        self.result_var = tk.StringVar()
        self.predict_button = tk.Button(self, text="Run", background="#006400", foreground="white",
                                        font=("Arial", 12), command=self.run_model)
        self.predict_button.pack(pady=10)
        self.predict_button.bind("<Enter>", on_enter)
        self.predict_button.bind("<Leave>", on_leave)

    def is_numeric(self, feature):
        dtype = self.controller.loaded_dtypes.get(feature, "")
        return dtype in ("int64", "float64")

    def get_unique_values(self, feature):
        return self.controller.loaded_unique_values.get(feature, [])

    def load_fields(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.entry_vars.clear()

        feature_names = self.controller.loaded_features
        target_name = self.controller.loaded_target

        method = self.controller.selected_method.get()
        self.title_label.config(text=f"Run {method} with Loaded Model")

        for i, feature in enumerate(feature_names):
            is_numeric = self.is_numeric(feature)
            
            ttk.Label(self.input_frame, text=feature, font=("Arial", 11), background="#F0F0F0")\
                .grid(row=0, column=i, padx=5, pady=5)
            var = tk.StringVar()
            self.entry_vars.append(var)

            if is_numeric:
                ttk.Entry(self.input_frame, textvariable=var, width=15)\
                    .grid(row=1, column=i, padx=5, pady=5)
            else:
                unique_values = self.controller.get_unique_values(feature)
                if unique_values:
                    var.set(unique_values[0])  # Set default explicitly
                option_menu = ttk.OptionMenu(self.input_frame, var, unique_values[0], *unique_values)
                option_menu.grid(row=1, column=i, padx=30, pady=5)

        self.result_label = ttk.Label(self, text=target_name + " (Predicted)", font=("Arial", 11, "italic"), background="#F0F0F0")
        self.result_label.pack(pady=(10, 0))
        self.result_entry = ttk.Entry(self, textvariable=self.result_var, font=("Arial", 12), state="readonly", width=30)
        self.result_entry.pack(pady=(0, 20))

    def run_model(self):
        try:
            model = self.controller.loaded_model
            method = self.controller.selected_method.get()

            input_values = []
            encoders = self.controller.loaded_encoders
            for i, feature in enumerate(self.controller.loaded_features):
                val = self.entry_vars[i].get()
                if feature in encoders:
                    val = encoders[feature].transform([val])[0]
                else:
                    val = float(val)
                input_values.append(val)
            
            input_df = pd.DataFrame([input_values], columns=self.controller.loaded_features)
            result = model.predict(input_df)[0]

            if self.controller.selected_method.get() == "Classification":
                target = self.controller.loaded_target
                encoders = self.controller.loaded_encoders
                if target in encoders:
                    result = encoders[target].inverse_transform([int(result)])[0]
                self.result_var.set(str(result))
            else:
                # For Regression or others, just display the number
                self.result_var.set(f"{result:.2f}")
        
        except ValueError:
            self.result_var.set("Invalid input.")
        except Exception as e:
            self.result_var.set(f"Error: {str(e)}")


if __name__ == "__main__":
    app = MiningApp()
    app.mainloop()

