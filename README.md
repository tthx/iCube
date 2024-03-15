# A propos du stage au laboratoire iCube, dans le cadre du dispositif *Respiration* de Orange

## Remerciements

Ils vont à M. [**Vincent LOECHNER**](http://icps.u-strasbg.fr/people/loechner/public_html/) et aux autres membres du laboratoire [**iCube**](https://icube.unistra.fr/).

## Objectifs

L'objectif est d'optimiser une partie d'un composant utilisé par [*openCARP*](https://opencarp.org/): [*Ginkgo*](https://ginkgo-project.github.io/), une librairie mathématique; *Ginkgo* devant remplacer [*PETSc*](https://petsc.org/release/), l'actuelle librairie mathématique utilisée. *Ginkgo* et *PETSc* manipulent principalement des vecteurs et des matrices, creuses et denses. La manipulation de matrices denses ou de vecteurs est propice aux [*optimisations polyédriques*](https://polyhedral.info/).

Le travail a emprunté le plan suivant:

- Explorer les possibles optimisations par des méthodes polyédriques du solveur *Ginkgo* utilisé par *openCARP*.
- Étudier les optimisations d'*openCARP* pour les architectures [*Exascale*](https://fr.wikipedia.org/wiki/Supercalculateur_exaflopique).

L'essentiel du travail a été porté sur une mise à jour des connaissances par les lectures de nombreux articles, de cours et, évidemment, des discussions avec M. Vincent LOECHNER, le directeur de ce stage. Les implémentations et leurs tests, en proportion, ont occupé quelques 20% des activités.

## Optimisations du solveur [*GC*](https://fr.wikipedia.org/wiki/M%C3%A9thode_du_gradient_conjugu%C3%A9) de *Ginkgo*

### Optimisations polyédriques

Le but est d'optimiser automatiquement un code: un outil (e.g. un compilateur) y identifie des sections appelées *Static Control Parts* ([*SCoP*](http://web.cs.ucla.edu/~pouchet/software/polyopt/doc/htmltexinfo/Specifics-of-Polyhedral-Programs.html)) pour y appliquer des transformations polyédriques qui optimisent ce code. Un programmeur, selon l'outil utilisé, peut être amené à guider l'identification des *SCoP* en posant des balises (e.g. des [*pragma*](https://gcc.gnu.org/onlinedocs/cpp/Pragmas.html)) dans le code à optimiser, et l'outil d'optimisation vérifie si la section du code balisé correspond à un *SCoP* avant d'y appliquer des transformations polyédriques.

#### Présentation générale du code de *Ginkgo*

*Ginkgo* est clairement structuré: les interfaces et leurs implémentations optimisées pour GPUs (supportant actuellement [*CUDA*](https://docs.nvidia.com/cuda/doc/index.html), [*HIP*](https://rocm.docs.amd.com/projects/HIP/en/latest/) et [*SYCL*](https://www.khronos.org/sycl/)) et CPUs par [*OpenMP*](https://www.openmp.org/) sont aisément identifiables. Une implémentation distribuée, via [*MPI*](https://en.wikipedia.org/wiki/Message_Passing_Interface), est en cours de développement. *Ginkgo* fournit aussi une implémentation dite de *référence*: c'est une version non optimisée séquentielle permettant de valider des algorithmes et n'est pas destinée à l'utilisation. *Note*: Par le choix pour la diversité des implémentations, afin d'exploiter au mieux les architectures des processeurs, *Ginkgo* a un volume de code assez important à maintenir, avec, cependant, l'avantage de cibler finement les modifications.

#### Présentation générale du code du solveur *GC* de *Ginkgo*

Le code commun à toutes les implémentations du *GC* de *Ginkgo* est disponible ici: [`ginkgo/core/solver/cg.cpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/core/solver/cg.cpp). La fonction

```cpp
template <typename ValueType>
template <typename VectorType>
void Cg<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x)
```

concentre l'ensemble des calculs, décomposés en trois étapes:

- l'initialisation, par la fonction

```cpp
#define GKO_DECLARE_CG_INITIALIZE_KERNEL(_type)                          \
void initialize(std::shared_ptr<const DefaultExecutor> exec,             \
                const matrix::Dense<_type>* b, matrix::Dense<_type>* r,  \
                matrix::Dense<_type>* z, matrix::Dense<_type>* p,        \
                matrix::Dense<_type>* q, matrix::Dense<_type>* prev_rho, \
                matrix::Dense<_type>* rho,                               \
                array<stopping_status>* stop_status)
```

- une boucle sur deux fonctions:

```cpp
#define GKO_DECLARE_CG_STEP_1_KERNEL(_type)                         \
void step_1(std::shared_ptr<const DefaultExecutor> exec,            \
            matrix::Dense<_type>* p, const matrix::Dense<_type>* z, \
            const matrix::Dense<_type>* rho,                        \
            const matrix::Dense<_type>* prev_rho,                   \
            const array<stopping_status>* stop_status)
```

et

```cpp
#define GKO_DECLARE_CG_STEP_2_KERNEL(_type)                               \
void step_2(std::shared_ptr<const DefaultExecutor> exec,                  \
            matrix::Dense<_type>* x, matrix::Dense<_type>* r,             \
            const matrix::Dense<_type>* p, const matrix::Dense<_type>* q, \
            const matrix::Dense<_type>* beta,                             \
            const matrix::Dense<_type>* rho,                              \
            const array<stopping_status>* stop_status)
```

Ces codes sont dans le fichier [`ginkgo/core/solver/gc_kernels.hpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/core/solver/cg_kernels.hpp) et définissent les interfaces des implémentions pour *CUDA*, *HIP*, *OpenMP*, etc.

**Note**: *openCARP* travaille principalement sur des matrices creuses, pourtant implémentées dans *Ginkgo*, or les fonctions `initialize`, `step_1` et `step_2` du *GC* de *Ginkgo* portent sur des matrices denses: les matrices creuses sont converties en matrices denses (*Ginkgo* impose et implémente des [*constructeurs*](https://en.wikipedia.org/wiki/Constructor_(object-oriented_programming)) de conversion dans tous les sens), ce qui implique que les matrices denses obtenues contiennent potentiellement un nombre conséquent de zéro et occupent un espace mémoire important...

Ainsi, les éléments à optimiser sont identifiés:

- Les fonctions `initialize`, `step_1` et `step_2` du *GC* de *Ginkgo*, et
- Les fonctions de la classe des matrices denses: [`ginkgo/core/matrix/dense_kernels.hpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/core/matrix/dense_kernels.hpp).

C'est sur l'implémentation dite de *référence* de *Ginkgo*

- des trois méthodes du *GC* de *Ginkgo*, [`ginkgo/reference/solver/cg_kernels.cpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/solver/cg_kernels.cpp), et
- des matrices denses, [`ginkgo/reference/matrix/dense_kernels.cpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/matrix/dense_kernels.cpp),

que se sont portées les *optimisations polyédriques*. Ces optimisations se limitent, actuellement, aux CPUs et ont été comparées à celles effectuées avec *OpenMP*, servant de référence.

Deux outils ont été utilisés:

- [*LLVM/Polly*](https://polly.llvm.org/) et
- [*PLUTO*](https://pluto-compiler.sourceforge.net/).

#### LLVM/Polly

Le travail avec *LLVM/Polly* est des plus simples et constitue l'idéal promu par les concepteurs des *optimisations polyédriques*: le seul travail du développeur se borne à lancer la compilation, en activant les options nécessaires, avec, au pire, comme modification de son code l'ajout de balises pour aider le compilateur à identifier les *SCoP*. La documentation de *LLVM/Polly* indique les options à utiliser pour le compilateur *[C](https://en.cppreference.com/w/c)/[C++](https://en.cppreference.com/w/)* [*Clang*](https://clang.llvm.org/): [*Using Polly with Clang*](https://polly.llvm.org/docs/UsingPollyWithClang.html).

Les options de *LLVM/Polly* que nous avons utilisées sont:

```
-Wno-unused-command-line-argument -mllvm -polly -mllvm -polly-dependences-computeout=0 -mllvm -polly-vectorizer=stripmine -mllvm -polly-parallel
```

Qu'il faut ajouter au fichier [`ginkgo/reference/CMakeLists.txt`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/CMakeLists.txt) par l'instruction:

```cmake
target_compile_options(ginkgo_reference PRIVATE "SHELL:-Wno-unused-command-line-argument -mllvm -polly -mllvm -polly-dependences-computeout=0 -mllvm -polly-vectorizer=stripmine -mllvm -polly-parallel")
```

**Notes**:

- Utiliser *LLVM/Polly* sur les implémentations optimisées pour *CUDA* ou *OpenMP* provoque des erreurs à l'exécution (mais la compilation avec *LLVM/Polly* activé se déroule comme un charme).
- Pour connaître les *SCoP* identifiés par *LLVM/Polly*, ajouter les options `-mllvm -polly-export`, décrit comme:

  > *Polly - Export Scops as JSON (Writes a .jscop file for each Scop)*

  Mais cette option plante *Clang* pour ses versions 15.x: pour l'utiliser, il faut passer aux versions strictement supérieures à 15.x de *Clang*. De plus, cette option gère mal les noms des fichiers générés: il faut l'appliquer avec parcimonie...

- *LLVM/Polly* a l'avantage d'être indépendant du langage de programmation utilisé par le programmeur: *LLVM/Polly* travaille sur les *représentations intermédiaires* ou [*IR*](https://en.wikipedia.org/wiki/Intermediate_representation) générées par le compilateur, responsable de la validation de la syntaxe.
- Certaines fonctions impliquées dans le solveur *GC* de *Ginkgo* ne respectant pas les critères du *SCoP* (e.g. `step_1` et `step_2`), ne pouvant donc pas être optimisées par *LLVM/Polly* ont été reformulées: voire la section [PLUTO](#pluto) sur la fonction `step_1` pour plus de détail.
- *LLVM/Polly* n'est qu'un des projets de [*The LLVM Compiler Infrastructure*](https://llvm.org/).
- *LLVM/Polly* passe par *OpenMP* pour la parallélisation et la vectorisation.

#### PLUTO

*PLUTO* représente probablement l'état de l'art des *optimisations polyédriques*: *PLUTO* regroupe la majorité des outils développés, donc des recherches, sur le sujet. Ses caractéristiques sont:

- L'utilisateur doit baliser (avec les directives `#pragma scop` et `#pragma endscop`) son code pour instruire *PLUTO* des *SCoP*.
- *PLUTO* est un compilateur *source-à-source*.
- *PLUTO* travaille sur le code source, ce qui le rend dépendant de la syntaxe du langage de programmation utilisé. Pour l'analyse syntaxique, *PLUTO* offre le choix entre [*Clan*](https://icps.u-strasbg.fr/~bastoul/development/clan/docs/clan.html) et [*Pet*](https://repo.or.cz/w/pet.git). *Pet* s'appuie sur *Clang* et offre un support complet du *C/C++* alors que *Clan* se limite à un *C* bridé. Cependant, l'utilisation de *Pet* par *PLUTO*, vers juin 2023, était impossible: ça ne fonctionnait tout simplement pas. **Note**: Alors que j'essayais d'améliorer l'intégration de *Pet* dans *PLUTO*, les auteurs de *PLUTO*, après des années d'inactivité sur leur [*github*](https://github.com/bondhugula/pluto), ont entrepris, entre autre, les mêmes travaux. [Une version stable](https://github.com/bondhugula/pluto/releases/download/0.12.0/pluto-0.12.0.tar.gz) de *PLUTO* est disponible depuis novembre 2023. Je ne l'ai pas, encore, testée.
- L'utilisation de *PLUTO* est un peu fastidieuse:

  * Extraire le code à optimiser et le traduire dans un *C* supporté par *PLUTO*,
  * Réintégrer, en l'adaptant à la syntaxe supportée par code d'origine, le code optimisé généré par *PLUTO*.

  Deux exemples:

  * Pour la fonction [`add_scaled`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/matrix/dense_kernels.cpp#L204), il fallait traduire en:

    ```c
    #include <stdlib.h>
    #include <math.h>
    void add_scaled(
      const size_t x_size_0,
      const size_t x_size_1,
      const size_t x_stride,
      const size_t y_stride,
      const double_t *alpha_values,
      const double_t **x_values,
      double_t **y_values) {
      size_t i, j;
      if (alpha->get_size()[1] == 1) {
        const auto valpha = alpha_values[0];
    #pragma scop
        for (i = 0; i < x_size_0; ++i)
          for (j = 0; j < x_size_1; ++j)
            y_values[i][j] += valpha * x_values[i][j];
    #pragma endscop
      } else {
    #pragma scop
        for (i = 0; i < x_size_0; ++i)
          for (j = 0; j < x_size_1; ++j)
            y_values[i][j] += alpha_values[j] * x_values[i][j];
    #pragma endscop
      }
    }
    ```

    La traduction est fastidieuse mais la sémantique du code est préservée.

  * La fonction [`step_1`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/solver/cg_kernels.cpp#L78C6-L78C12) est plus problématique: elle ne respecte pas les caractéristiques du *SCoP* à cause et de l'utilisation des tests `if` et de l'instruction non-affine

    ```c
    p->at(i, j) = z->at(i, j) + tmp * p->at(i, j);
    ```

    où:

    ```c
    tmp = rho->at(j) / prev_rho->at(j);
    ```

    Pour que `step_1` rentre dans les cases du *SCoP*, il faut utiliser une structure temporaire et traduire `step_1` en:

    ```c
    #include <stdlib.h>
    #include <math.h>
    void step_1(
        const size_t p_size_0,
        const size_t p_size_1,
        const size_t rho_size_1,
        const size_t prev_rho_size_1,
        const double_t **prev_rho_values,
        const double_t **rho_values,
        const double_t **z_values,
        double_t **p_values) {
        size_t i, j;
        double_t *tmp = malloc(sizeof(double_t[p_size_1]));
        for (j = 0; j < p_size_1; ++j)
            tmp[j] = rho_values[j / rho_size_1][j % rho_size_1] /
                prev_rho_values[j / prev_rho_size_1][j % prev_rho_size_1];
    #pragma scop
        for (i = 0; i < p_size_0; ++i)
            for (j = 0; j < p_size_1; ++j)
                p_values[i][j] = z_values[i][j] + tmp[j] * p_values[i][j];
    #pragma endscop
        free(tmp);
    }
    ```
- *Ginkgo* dispose heureusement d'une batterie de test, ce qui a facilité la validation des *optimisations polyédriques* par *PLUTO*.
- *PLUTO* passe par *OpenMP* pour la parallélisation et la vectorisation.

#### Résultats

Ce fut implacable: ni *LLVM/Polly*, ni *PLUTO* n'ont pu, et de très loin, rivaliser avec l'implémentation du solveur *GC* optimisée avec *OpenMP* de *Ginkgo*. C'est probablement définitif: les codes de *Ginkgo* supportant les contraintes du *SCoP* ne contiennent pas de dépendances complexe. Il est alors quelque peu rassurant de constater que des processus automatiques n'ont pas pu dépasser les capacités humaines, du moins celles des concepteurs de *Ginkgo*.

### Algorithmiques

Constatant que la majorité des opérations impliquées dans le solveur *CG* de *Ginkgo* peut être optimisée si les matrices denses sont traitées par colonne plutôt que par ligne, une reformulation algorithmique a été effectuée. Tout en sachant que *Ginkgo* stocke ses matrices denses par ligne, et, donc, les parcourir par colonne dégraderont fortement les performances à cause des défauts de caches. Ces dégradations sont amplifiées lorsque les traitements sont parallélisés. **Note**: Pour éviter le problème de défaut de caches, il faut ajouter des opérations de transpositions coûteuses en temps de calcul et en espace mémoire; une option inenvisageable lorsque le but est d'optimiser.

Lors de la reformulation, en exploitant les propriétés algébriques des opérations d'addition et de multiplication sur les éléments neutres (i.e. `0` et `1`) par l'ajout de tests (mais un code plus complexe), nous avons évité de lancer des calculs sur de potentiels vastes volumes de données.

Par exemple:

- La fonction [`compute_norm2`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/matrix/dense_kernels.cpp#L347) a été reformulée en:

  ```cpp
  template <typename ValueType>
  void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Dense<ValueType>* x,
                     matrix::Dense<remove_complex<ValueType>>* result,
                     array<char>&)
  {
  #pragma omp parallel for
      for (size_type j = 0; j < x->get_size()[1]; ++j) {
          auto val = zero<remove_complex<ValueType>>();
  #pragma omp simd reduction(+ : val)
          for (size_type i = 0; i < x->get_size()[0]; ++i) {
              val += std::norm(x->at(i, j));
          }
          result->at(0, j) = std::sqrt(val);
      }
  }
  ```

- La fonction [`step_1`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/solver/cg_kernels.cpp#L78) en:

  ```cpp
  template <typename ValueType>
  void step_1(std::shared_ptr<const DefaultExecutor> exec,
              matrix::Dense<ValueType>* p, const matrix::Dense<ValueType>* z,
              const matrix::Dense<ValueType>* rho,
              const matrix::Dense<ValueType>* prev_rho,
              const array<stopping_status>* stop_status)
  {
  #pragma omp parallel for
      for (size_type j = 0; j < p->get_size()[1]; ++j) {
          if (!stop_status->get_const_data()[j].has_stopped()) {
              const auto prev_rho_value = prev_rho->at(j);
              auto val = zero<ValueType>();
              if (is_nonzero(prev_rho_value)) {
                  const auto rho_value = rho->at(j);
                  if (is_nonzero(rho_value)) {
                      val = rho_value / prev_rho_value;
                  }
              }
              if (is_zero(val)) {
  #pragma omp simd
                  for (size_type i = 0; i < p->get_size()[0]; ++i) {
                      p->at(i, j) = z->at(i, j);
                  }
              } else {
                  if (val != one<ValueType>()) {
  #pragma omp simd
                      for (size_type i = 0; i < p->get_size()[0]; ++i) {
                          p->at(i, j) = z->at(i, j) + (val * p->at(i, j));
                      }
                  } else {
  #pragma omp simd
                      for (size_type i = 0; i < p->get_size()[0]; ++i) {
                          p->at(i, j) += z->at(i, j);
                      }
                  }
              }
          }
      }
  }
  ```

- La fonction [`step_2`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/solver/cg_kernels.cpp#L103) en:

  ```cpp
  template <typename ValueType>
  void step_2(std::shared_ptr<const DefaultExecutor> exec,
              matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
              const matrix::Dense<ValueType>* p,
              const matrix::Dense<ValueType>* q,
              const matrix::Dense<ValueType>* beta,
              const matrix::Dense<ValueType>* rho,
              const array<stopping_status>* stop_status)
  {
  #pragma omp parallel for
      for (size_type j = 0; j < p->get_size()[1]; ++j) {
          if (!stop_status->get_const_data()[j].has_stopped()) {
              const auto rho_value = rho->at(j);
              auto val = zero<ValueType>();
              if (is_nonzero(rho_value)) {
                  const auto beta_value = beta->at(j);
                  if (is_nonzero(beta_value)) {
                      val = rho_value / beta_value;
                  }
              }
              if (is_nonzero(val)) {
                  if (val != one<ValueType>()) {
  #pragma omp simd
                      for (size_type i = 0; i < p->get_size()[0]; ++i) {
                          x->at(i, j) += val * p->at(i, j);
                          r->at(i, j) -= val * q->at(i, j);
                      }
                  } else {
  #pragma omp simd
                      for (size_type i = 0; i < p->get_size()[0]; ++i) {
                          x->at(i, j) += p->at(i, j);
                          r->at(i, j) -= q->at(i, j);
                      }
                  }
              }
          }
      }
  }
  ```

#### Résultats

Avec les outils de mesure de performances fournis par *Ginkgo*:

- Les versions séquentielles reformulées de:

  * `compute_norm2_dispatch` est `2,158910204x` plus rapide,
  * `compute_conj_dot_dispatch` est `2,059145506x` plus rapide,
  * `step_1` est `14,473490735x` plus rapide,
  * `step_2` est `7,1691806x` plus rapide

  que leurs versions séquentielles de référence.

- Les versions reformulées optimisées avec *OpenMP*:

  * `compute_norm2_dispatch` est `3,526641558x` plus lente,
  * `compute_conj_dot_dispatch` est `2,508054004x` plus lente,
  * `step_1` est `1,261263298x` plus lente,
  * `step_2` est `1,459083835x` plus lente

  que leurs versions optimisées avec *OpenMP* de référence.

En dépit des défauts de caches causés par les parcours des matrices denses par colonne alors qu'elles sont stockées par ligne.

**Notes**:

- Les versions séquentielles reformulées de `step_1` et `step_2` ne sont, respectivement, que `1,233304792x` et `1,289617482x` plus lentes que leurs versions optimisées par *OpenMP* de référence.
- Les concepteurs de *Ginkgo*, depuis 2021, songent à fournir plusieurs modes de stockage des matrices denses: [Clarify the behavioral differences between a dense matrix and a multivector](https://github.com/ginkgo-project/ginkgo/issues/796). Le débat est encore ouvert.

## Optimisations d'openCARP

### Exploiter les optimisations pour *GPUs* par *PETSc*

*openCARP* impose des interfaces aux

- [vecteurs](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/fem/slimfem/src/SF_abstract_vector.h),
- [matrices](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/fem/slimfem/src/SF_abstract_matrix.h) et
- [solveurs](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/fem/slimfem/src/SF_abstract_matrix.h).

Ces interfaces sont implémentées soit avec *PETSc*, soit avec *Ginkgo*. L'implémentation d'*openCARP* avec *PETSc* n'exploite actuellement que les *CPUs*. Cependant, *PETSc* dispose, au moins pour les matrices et les vecteurs, des implémentations exploitant les *GPUs*: [*GPU Support Roadmap*](https://petsc.org/release/overview/gpu_roadmap/).

Trois implémentations ont été effectuées pour tester les offres *GPUs* de *PETSc*:

- Une implémentation spécifique pour *CUDA*,
- une implémentation spécifique pour [*Kokkos*](https://kokkos.org/),
- une implémentation générique où il est possible de sélectionner et le type vecteur et le type de matrice à la ligne de commande. Les types de vecteur supportés sont présentés ici: [`VecType`](https://petsc.org/release/manualpages/Vec/VecType/); et les types de matrices supportées sont présentés là: [`MatType`](https://petsc.org/main/manualpages/Mat/MatType/). Pour utiliser l'implémentation générique, il faut ajouter à la ligne de commande d'*openCARP*, par exemple pour avoir des vecteurs [`cuda`](https://petsc.org/release/manualpages/Vec/VECCUDA/) et des matrices [`aijcusparse`](https://petsc.org/main/manualpages/Mat/MATAIJCUSPARSE/):

```shell
+ \
-mat_type aijcusparse \
-vec_type cuda
```

Les trois implémentations ont été validées en effectuant les tests dits de [*régression*](https://git.opencarp.org/openCARP/experiments/-/blob/master/TESTS.md) fournis par *openCARP*.

Contrairement à ce qui est attendu, les résultats ont montré de fortes dégradations (parfois d'un facteur `10x`) des performances d'*openCARP* lorsque l'exécution sur *GPUs* de *PETSc* est utilisée. En utilisant les options de [*profiling*](https://petsc.org/release/manual/profiling/) de *PETSc*, par exemple sur l'outil [`bench.cc`](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/physics/limpet/src/bench.cc) d'*openCARP* pour mesurer les performances des *modèles ioniques* et qui n'utilise de *PETSc* que les vecteurs:

```shell
#-------------------------------------------------------------------------------
#                             bench: Plugin ISAC_Hu
#-------------------------------------------------------------------------------

/opt/openCARP/master/embedded/mpich/generic/bin/bench \
  --imp=LuoRudy91 \
  --plug-in=ISAC_Hu \
  --fout=ISAC_Hu/ISAC_Hu \
  --bin \
  --no-trace \
  --validate \
  --duration 1000 + \
  -log_view \
  -log_view_memory \
  -log_view_gpu_time \
  -mat_type aijcusparse \
  -vec_type cuda


*** GIT tag:              386fad2
*** GIT hash:             386fad226eed145b2c3e9a7caeeca5fea0f9789f
*** GIT repo:             https://git.opencarp.org/openCARP/openCARP.git
*** dependency commits:

Plug-in: ISAC_Hu

Outputting the following quantities at each time:
      Time          Vm      Lambda        Iion

Running simulation on target mlir-cuda



All done!


setup time          0.074113 s
initialization time 0.017918 s
main loop time      3.388433 s
total ode time      3.362835 s

mn/avg/mx loop time 0.026053 0.033884 61.257151 ms
mn/avg/mx ODE time  0.025972 0.033628 61.202800 ms
real time factor    0.297368
```

Les options de *profiling* de *PETSc* nous indiquent:

```shell
------------------------------------------------------------------
 PETSc Performance Summary:
------------------------------------------------------------------



      ##########################################################
      #                                                        #
      #                       WARNING!!!                       #
      #                                                        #
      #   This code was run with -log_view_gpu_time            #
      #   This provides accurate timing within the GPU kernels #
      #   but can slow down the entire computation by a        #
      #   measurable amount. For fastest runs we recommend     #
      #   not using this option.                               #
      #                                                        #
      ##########################################################


/opt/openCARP/master/embedded/mpich/generic/bin/bench on a  named 55a3a2a1c071 with 1 processor, by Unknown Mon Mar  4 16:27:44 2024
Using Petsc Release Version 3.20.5, unknown

                         Max       Max/Min     Avg       Total
Time (sec):           3.487e+00     1.000   3.487e+00
Objects:              0.000e+00     0.000   0.000e+00
Flops:                1.000e+02     1.000   1.000e+02  1.000e+02
Flops/sec:            2.867e+01     1.000   2.867e+01  2.867e+01
Memory (bytes):       1.051e+05     1.000   1.051e+05  1.051e+05
MPI Msg Count:        0.000e+00     0.000   0.000e+00  0.000e+00
MPI Msg Len (bytes):  0.000e+00     0.000   0.000e+00  0.000e+00
MPI Reductions:       0.000e+00     0.000

Flop counting convention: 1 flop = 1 real number operation of type (multiply/divide/add/subtract)
                            e.g., VecAXPY() for real vectors of length N --> 2N flops
                            and VecAXPY() for complex vectors of length N --> 8N flops

Summary of Stages:   ----- Time ------  ----- Flop ------  --- Messages ---  -- Message Lengths --  -- Reductions --
                        Avg     %Total     Avg     %Total    Count   %Total     Avg         %Total    Count   %Total
 0:      Main Stage: 3.4875e+00 100.0%  1.0000e+02 100.0%  0.000e+00   0.0%  0.000e+00        0.0%  0.000e+00   0.0%

------------------------------------------------------------------------------------------------------------------------
See the 'Profiling' chapter of the users' manual for details on interpreting output.
Phase summary info:
   Count: number of times phase was executed
   Time and Flop: Max - maximum over all processors
                  Ratio - ratio of maximum to minimum over all processors
   Mess: number of messages sent
   AvgLen: average message length (bytes)
   Reduct: number of global reductions
   Global: entire computation
   Stage: stages of a computation. Set stages with PetscLogStagePush() and PetscLogStagePop().
      %T - percent time in this phase         %F - percent flop in this phase
      %M - percent messages in this phase     %L - percent message lengths in this phase
      %R - percent reductions in this phase
   Total Mflop/s: 10e-6 * (sum of flop over all processors)/(max time over all processors)
   Memory usage is summed over all MPI processes, it is given in mega-bytes
   Malloc Mbytes: Memory allocated and kept during event (sum over all calls to event). May be negative
   EMalloc Mbytes: extra memory allocated during event and then freed (maximum over all calls to events). Never negative
   MMalloc Mbytes: Increase in high water mark of allocated memory (sum over all calls to event). Never negative
   RMI Mbytes: Increase in resident memory (sum over all calls to event)
   GPU Mflop/s: 10e-6 * (sum of flop on GPU over all processors)/(max GPU time over all processors)
   CpuToGpu Count: total number of CPU to GPU copies per processor
   CpuToGpu Size (Mbytes): 10e-6 * (total size of CPU to GPU copies per processor)
   GpuToCpu Count: total number of GPU to CPU copies per processor
   GpuToCpu Size (Mbytes): 10e-6 * (total size of GPU to CPU copies per processor)
   GPU %F: percent flops on GPU in this event
------------------------------------------------------------------------------------------------------------------------
Event                Count      Time (sec)     Flop                              --- Global ---  --- Stage ----  Total  Malloc EMalloc MMalloc RMI   GPU    - CpuToGpu -   - GpuToCpu - GPU
                   Max Ratio  Max     Ratio   Max  Ratio  Mess   AvgLen  Reduct  %T %F %M %L %R  %T %F %M %L %R Mflop/s Mbytes Mbytes Mbytes Mbytes Mflop/s Count   Size   Count   Size  %F
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--- Event Stage 0: Main Stage

PetscBarrier           2 1.0 3.9924e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
VecSet              1106 1.0 5.0963e-02 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  1  0  0  0  0   1  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
VecCUDACopyTo        100 1.0 1.6782e-03 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0     0       0       0       0       0    100 8.00e-04    0 0.00e+00  0
VecCUDACopyFrom     1203 1.0 2.0490e-02 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  1  0  0  0  0   1  0  0  0  0     0     0       0       0       0       0      0 0.00e+00 1203 9.62e-03  0
DCtxCreate             2 1.0 4.6355e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
DCtxDestroy            2 1.0 4.7569e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
DCtxSetUp              2 1.0 4.0234e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
DCtxSetDevice          2 1.0 4.7570e-05 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
DCtxSync            2409 1.0 2.9572e-02 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00  1  0  0  0  0   1  0  0  0  0     0     0       0       0       0       0      0 0.00e+00    0 0.00e+00  0
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Object Type          Creations   Destructions. Reports information only for process 0.

--- Event Stage 0: Main Stage

           Container     4              4
              Vector     6              6
  PetscDeviceContext     2              0
========================================================================================================================
Average time to get PetscTime(): 2.28e-08
#PETSc Option Table entries:
-log_view # (source: command line)
-log_view_gpu_time # (source: command line)
-log_view_memory # (source: command line)
-mat_type aijcusparse # (source: command line)
-options_left no # (source: code)
-vec_type cuda # (source: command line)
#End of PETSc Option Table entries
Compiled without FORTRAN kernels
Compiled with full precision matrices (default)
sizeof(short) 2 sizeof(int) 4 sizeof(long) 8 sizeof(void*) 8 sizeof(PetscScalar) 8 sizeof(PetscInt) 4
-----------------------------------------
Libraries compiled on 2024-03-03 10:37:33 on 55a3a2a1c071
Machine characteristics: Linux-6.8.0-7-lowlatency-x86_64-with-glibc2.35
Using PETSc directory: /opt/petsc/release/embedded/mpich
Using PETSc arch:
-----------------------------------------

Using C compiler: /opt/petsc/release/embedded/mpich/bin/mpicc -O3 -march=native -O3 -march=native
Using Fortran compiler: /opt/petsc/release/embedded/mpich/bin/mpif90 -O3 -march=native -O3 -march=native
-----------------------------------------

Using include paths: -I/opt/petsc/release/embedded/mpich/include -I/usr/local/cuda-11.8/include
-----------------------------------------

Using C linker: /opt/petsc/release/embedded/mpich/bin/mpicc
Using Fortran linker: /opt/petsc/release/embedded/mpich/bin/mpif90
-----------------------------------------
```

Nous constatons que *PETSc* passe un temps conséquent à effectuer des copies entre *CPU* et *GPU*. Après analyse, ces copies résultent de l'utilisation de *PETSc* par *openCARP* et non d'un fonctionnement interne à *PETSc*:

- L'interface des vecteurs d'*openCARP* propose quatre fonctions:

  * `ptr`, implémentée avec *PETSc* par [`VecGetArray`](https://petsc.org/release/manualpages/Vec/VecGetArray/),
  * `release_ptr`, implémentée avec *PETSc* par [`VecRestoreArray`](https://petsc.org/release/manualpages/Vec/VecRestoreArray/),
  * `const_ptr`, implémentée avec *PETSc* par [`VecGetArrayRead`](https://petsc.org/release/manualpages/Vec/VecGetArrayRead/),
  * `const_release_ptr`, implémentée avec *PETSc* par [`VecRestoreArrayRead`](https://petsc.org/release/manualpages/Vec/VecRestoreArrayRead/).

  `VecGetArray` a pour fonction de:

  > *Returns a pointer to a contiguous array that contains this MPI processes’s portion of the vector data*

  et

  > *For the standard PETSc vectors, [`VecGetArray()`](https://petsc.org/release/manualpages/Vec/VecGetArray/) returns a pointer to the local data array and does not use any copies. If the underlying vector data is not stored in a contiguous array this routine will copy the data to a contiguous array and return a pointer to that. You MUST call [`VecRestoreArray()`](https://petsc.org/release/manualpages/Vec/VecRestoreArray/) when you no longer need access to the array.*

  *openCARP* propose ces fonctions car, par exemple dans les *modèles ioniques*, les calculs ne sont pas effectués avec les vecteurs instanciés de la classe abstraite [`abstract_vector`](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/fem/slimfem/src/SF_abstract_vector.h#L54) implémentée avec la classe [`petsc_vector`](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/numerics/petsc/SF_petsc_vector.h), mais avec des vecteurs de la classe [`vector`](https://git.opencarp.org/openCARP/openCARP/-/blob/5649e7b4f0aa0b9f676e15505e2d98183c4715df/fem/slimfem/src/SF_vector.h): les `petsc_vector` servent principalement de tampon avec les `vector`. Comme les vecteurs *PETSc* sont, en principe, en mémoire *GPU*, les appels à `VecGetArray` effectuent des copies du GPU vers le CPU; et `VecRestoreArray` fait le chemin contraire... Comme la documentation sur `VecGetArray` est assez imprécise, il faut consulter la documentation de [`VecGetArrayAndMemType`](https://petsc.org/release/manualpages/Vec/VecGetArrayAndMemType/) pour être au clair:

  > *Like [`VecGetArray()`](https://petsc.org/release/manualpages/Vec/VecGetArray/), but if this is a standard device vector (e.g., [`VECCUDA`](https://petsc.org/release/manualpages/Vec/VECCUDA/)), the returned pointer will be a device pointer to the device memory that contains this MPI processes’s portion of the vector data.*

  et

  > *Device data is guaranteed to have the latest value. Otherwise, when this is a host vector (e.g., [`VECMPI`](https://petsc.org/release/manualpages/Vec/VECMPI/)), this routine functions the same as [`VecGetArray()`](https://petsc.org/release/manualpages/Vec/VecGetArray/) and returns a host pointer.*
  >
  > *For [`VECKOKKOS`](https://petsc.org/release/manualpages/Vec/VECKOKKOS/), if Kokkos is configured without device (e.g., use serial or openmp), per this function, the vector works like [`VECSEQ`](https://petsc.org/release/manualpages/Vec/VECSEQ/)/[`VECMPI`](https://petsc.org/release/manualpages/Vec/VECMPI/); otherwise, it works like [`VECCUDA`](https://petsc.org/release/manualpages/Vec/VECCUDA/) or [`VECHIP`](https://petsc.org/release/manualpages/Vec/VECHIP/) etc.*
  >
  > *Use [`VecRestoreArrayAndMemType()`](https://petsc.org/release/manualpages/Vec/VecRestoreArrayAndMemType/) when the array access is no longer needed.*

  **Ces transferts sont imputables à la logique permise par l'interface des vecteurs *openCARP*: des composants constituant *openCARP* ont implémenté leur propre vecteur plutôt que de s'accorder sur une seule implémentation.**

  *Note*: La combinaison où les vecteurs sont de type `standard` (i.e. sur *CPU*) et les matrices de type `aijcusparse` (i.e. sur *GPU*) ne fonctionne pas.

- Cependant, les développeurs de *PETSc* n'ont probablement pas pris en compte ces problèmes de transfert *CPU*<->*GPU* dans toutes les fonctions qu'ils ont implémentées... Par exemple pour la fonction [`VecEqual`](https://petsc.org/release/src/vec/vec/utils/vinv.c.html#VecEqual):

```c
PetscErrorCode VecEqual(Vec vec1, Vec vec2, PetscBool *flg)
{
  const PetscScalar *v1, *v2;
  PetscInt           n1, n2, N1, N2;
  PetscBool          flg1;

  PetscFunctionBegin;
  PetscAssertPointer(flg, 3);
  if (vec1 == vec2) *flg = PETSC_TRUE;
  else {
    PetscCall(VecGetSize(vec1, &N1));
    PetscCall(VecGetSize(vec2, &N2));
    if (N1 != N2) flg1 = PETSC_FALSE;
    else {
      PetscCall(VecGetLocalSize(vec1, &n1));
      PetscCall(VecGetLocalSize(vec2, &n2));
      if (n1 != n2) flg1 = PETSC_FALSE;
      else {
        PetscCall(VecGetArrayRead(vec1, &v1));
        PetscCall(VecGetArrayRead(vec2, &v2));
        PetscCall(PetscArraycmp(v1, v2, n1, &flg1));
        PetscCall(VecRestoreArrayRead(vec1, &v1));
        PetscCall(VecRestoreArrayRead(vec2, &v2));
      }
    }
    /* combine results from all processors */
    PetscCall(MPIU_Allreduce(&flg1, flg, 1, MPIU_BOOL, MPI_MIN, PetscObjectComm((PetscObject)vec1)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

  Il faudra alors examiner les implémentations de chaque fonction *PETSc* utilisée...

### Vers le *Exascale*

Les défis qu'imposent la [*portabilité*](https://performanceportability.org/), le *multi-GPUs* et la hiérarchie mémoire pourraient être solubles par l'utilisation des [*algorithmes parallèles de la STL*](https://en.cppreference.com/w/cpp/algorithm), du *Global Address Space* et du *task-based programming*: [*HPx*](https://hpx.stellar-group.org/).

## Description de l'arborescence *github repository*

- `docker/openCARP` contient un ensemble de script:
  * `build-env.sh` définit les variables relatives aux:

    * informations *github* (*repository*, *branch*),
    * répertoires des sources et d'installation,
    * contextes de compilation, d'utilisation, etc.

    des composants utilisés par *openCARP* et de *openCARP*,

  * les fichiers préfixés par `build-` servent à compiler les composants utilisés par *openCARP*,
  * `get-cmake.sh` permet d'installer la version désirée de `cmake`. *Note*: les versions supérieures à `3.28.0` ne permettent pas à *openCARP* de détecter *CUDA*...
  * `get-ginkgo-data.sh` permet de télécharger toutes les matrices creuses dites de références (i.e. [*SuiteSparse Matrix Collection Formerly the University of Florida Sparse Matrix Collection*](https://sparse.tamu.edu/)) pour mesurer les performances de *Ginkgo* (cf. [*Running the benchmarks*](https://github.com/ginkgo-project/ginkgo/blob/master/BENCHMARKING.md)),
  * `ginkgo-benchmarks.sh` permet de sélectionner, en indiquant un intervalle de leur nombre d'élément non nul, les matrices creuses dites de références téléchargées à utiliser pour mesurer les performances de *Ginkgo*,
  * `openCARP-docker.sh` permet d'installer un *container* *docker* à partir d'une *image* de *Ubuntu* avec un ensemble de paramètre présélectionné (e.g. le *container* résultant est paramétré pour utiliser les cartes *NVIDIA* et exploiter la mémoire partagée - [*MPICH*](https://www.mpich.org/) nécessitant [*UCX*](https://openucx.org/)),
  * `openCARP-apt.sh` permet d'installer sur une distribution *Ubuntu* les paquets nécessaires pour compiler et exécuter les composants de *openCARP*,
  * `poly-apt.sh` permet de n'installer, sur une distribution *Ubuntu*, que les paquets nécessaires pour compiler et exécuter les composants de *PLUTO*,
  * `openCARP-example.sh` permet de faciliter l'exécution de l'exemple [*Basic usage of single cell tool bench - Limit cycle experiments*](https://opencarp.org/documentation/examples/01_ep_single_cell/01_basic_bench) avec les différentes modifications de *openCARP*,
  * `openCARP-regression.sh` permet d'exécuter les tests dites de [*régression*](https://git.opencarp.org/openCARP/experiments/-/blob/master/TESTS.md) de *openCARP*,
  * `openCARP-spack.sh` permet de compiler *openCARP* et ses composants à partir de [*spack*](https://spack.io/),
  * `openCARP-setup.sh` permet de compiler *openCARP* et ses composants sans environnement particulier,
  * `pluto-test.sh` permet d'exécuter la batterie de test fournie par *PLUTO*,
  * `runtime-env.sh` définit les variables d'environnement nécessaires pour exécuter *openCARP*.
- `ginkgo`
  * `develop` contient les codes relatives à la section [*Algorithmiques*](#algorithmiques),
  * `pluto` contient les codes relatives à la section [*PLUTO*](#pluto).
- `openCARP` contient les codes relatives à la section [*Exploiter les optimisations pour GPUs par PETSc*](#exploiter-les-optimisations-pour-gpus-par-petsc).
- Le reste correspond aux modifications, nécessaires pour les compiler, de sources des composants utilisés par *openCARP*.
