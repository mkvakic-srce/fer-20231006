# Aplikacije za strojno učenje na resursu Supek

  Ispod se nalaze skripte PBS i Python u svrhu održavanja radionice
  *Aplikacije za strojno učenje na resursu Supek*

## Opis

  Skripte se sastoje od parova skripta PBS i python koje demonstriraju razne
  knjižnice koje su u tom trenu bile dostupne na superračunalu Supek. Svaka
  knjižnica, ako je primjenjivo, napisana je u verziji koja koristi samo jedan
  procesor/čvor ili više njih.

  Opis:
  1. `tensorflow-singlegpu.*` - TensorFlow na jednom grafičkom procesoru
  1. `tensorflow-distributed.*` - TensorFlow na više grafičkih procesora i čvorova
  1. `pytorch-singlegpu.*` - PyTorch na jednom grafičkom procesoru
  1. `pytorch-distributed.*` - PyTorch na jednom više grafičkih procesora i čvorova
  1. `sklearn-threads.*` - Distribucija scikit-learna putem multi-threadinga
  1. `sklearn-dask.*` - Distribucija scikit-learna putem Daska
  1. `sklearn-dask_dask.*` - Distribucija scikit-learna za big data problem
  1. `pytorch-ray-train.*` - Distribucija PyTorcha korištenjem Ray Train
  1. `tensorflow-ray-tune.*` - Distribucija TensorFlowa korištenjem Ray Tune
