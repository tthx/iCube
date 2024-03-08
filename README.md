# A propos du stage au laboratoire iCube, dans le cadre du dispositif *Respiration* de Orange

## Remerciements

Ils vont à M. [**Vincent LOECHNER**](http://icps.u-strasbg.fr/people/loechner/public_html/) et aux autres membres du laboratoire [**iCube**](https://icube.unistra.fr/).

## Objectifs

L'objectif est d'optimiser une partie d'un composant utilisé par [*OpenCARP*](https://opencarp.org/): [*Ginkgo*](https://ginkgo-project.github.io/), une librairie mathématique; *Ginkgo* devant remplacer [*PETSc*](https://petsc.org/release/), l'actuelle librairie mathématique utilisée. *Ginkgo* et *PETSc* manipulent principalement des vecteurs et des matrices, et creuses et denses. La manipulation de matrices denses ou de vecteurs est propice aux [*optimisations polyédriques*](https://polyhedral.info/).

Le travail a emprunté le plan suivant:

- Optimiser par les méthodes polyédriques le solveur de *Ginkgo* utilisé par *OpenCARP*.
- Études des optimisations de *OpenCARP* pour les architectures [*Exascale*](https://fr.wikipedia.org/wiki/Supercalculateur_exaflopique).

L'essentiel du travail a été porté sur une mise à jour des connaissances par les lectures de nombreux articles, de cours et, évidemment, des discussions avec M. Vincent LOECHNER, le directeur de ce stage. Les implémentations et leurs tests, en proportion, ont occupé quelques 20% des activités.

## Optimisations du solveur [*GC*](https://fr.wikipedia.org/wiki/M%C3%A9thode_du_gradient_conjugu%C3%A9) de *Ginkgo*

### Optimisations polyédriques

Le but est d'optimiser automatiquement un code: un outil (e.g. un compilateur) y identifie des sections appelées *Static Control Parts* ([*SCoP*](http://web.cs.ucla.edu/~pouchet/software/polyopt/doc/htmltexinfo/Specifics-of-Polyhedral-Programs.html)) pour y appliquer des transformations polyédriques. Un programmeur, selon l'outil utilisé, peut être amené à guider l'identification des *SCoP* en posant des balises (e.g. des [*pragma*](https://gcc.gnu.org/onlinedocs/cpp/Pragmas.html)) dans le code à optimiser, l'outil d'optimisation vérifie si la section du code balisé correspond à un *SCoP* avant d'y appliquer des transformations polyédriques.

#### Présentation générale du code de *Ginkgo*

*Ginkgo* est clairement structuré: les interfaces et leurs implémentations optimisées pour GPUs (supportant actuellement [*CUDA*](https://docs.nvidia.com/cuda/doc/index.html), [*HIP*](https://rocm.docs.amd.com/projects/HIP/en/latest/) et [*SYCL*](https://www.khronos.org/sycl/)) et CPUs par [*OpenMP*](https://www.openmp.org/) sont aisément identifiables. Une implémentation distribuée, via [*MPI*](https://en.wikipedia.org/wiki/Message_Passing_Interface), est en cours de développement. *Ginkgo* fournit aussi une implémentation dite de *référence*: c'est une version séquentielle permettant de valider des algorithmes et n'est pas destinée à l'utilisation. *Note*: Par le choix pour la diversité des implémentations, afin d'exploiter au mieux les architectures des processeurs, *Ginkgo* a un volume de code conséquent à maintenir, avec, cependant, l'avantage de cibler finement les modifications.

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

**Note**: *OpenCARP* travaille principalement sur des matrices creuses, pourtant implémentées dans *Ginkgo*, or les fonctions `initialize`, `step_1` et `step_2` du *GC* de *Ginkgo* portent sur des matrices denses: les matrices creuses sont convertis en matrices denses (*Ginkgo* impose et implémente des [*constructeurs*](https://en.wikipedia.org/wiki/Constructor_(object-oriented_programming)) de conversion dans tous les sens), ce qui implique que les matrices denses obtenues contiennent potentiellement un nombre conséquent de zéro et occupent un espace mémoire en conséquence...

Ainsi, les éléments à optimiser sont identifiés:

- Les fonctions `initialize`, `step_1` et `step_2` du *GC* de *Ginkgo*, et
- Les fonctions de la classe des matrices denses: [`ginkgo/core/matrix/dense_kernels.hpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/core/matrix/dense_kernels.hpp).

C'est sur l'implémentation dite de *référence* de *Ginkgo*

- des trois méthodes du *GC* de *Ginkgo*, [`ginkgo/reference/solver/cg_kernels.cpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/solver/cg_kernels.cpp), et
- des matrices denses, [`ginkgo/reference/matrix/dense_kernels.cpp`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/matrix/dense_kernels.cpp),

que se sont portées les *optimisations polyédriques*. Ces optimisations se limitent, actuellement, aux CPUs et ont été comparées à celles effectuées avec *OpenMP*, servant de références.

Deux outils ont été utilisés:

- [*LLVM/Polly*](https://polly.llvm.org/) et
- [*PLUTO*](https://pluto-compiler.sourceforge.net/).

#### LLVM/Polly

Le travail avec *LLVM/Polly* est des plus simples et constitue l'idéal promu par les concepteurs des *optimisations polyédriques*: le seul travail du développeur se borne à lancer la compilation, en activant les options nécessaires, avec, au pire, comme modification de son code, l'ajout de balise, pour aider le compilateur à identifier les *SCoP*, pour optimiser son code. La documentation de *LLVM/Polly* instruit sur les options à utiliser pour le compilateur *[C](https://en.cppreference.com/w/c)/[C++](https://en.cppreference.com/w/)* [*Clang*](https://clang.llvm.org/): [*Using Polly with Clang*](https://polly.llvm.org/docs/UsingPollyWithClang.html).

Les options de *LLVM/Polly* utilisées sont:

```
-Wno-unused-command-line-argument -mllvm -polly -mllvm -polly-dependences-computeout=0 -mllvm -polly-vectorizer=stripmine -mllvm -polly-parallel
```

Qu'il faut ajouter au fichier [`ginkgo/reference/CMakeLists.txt`](https://github.com/ginkgo-project/ginkgo/blob/49242ff89af1e695d7794f6d50ed9933024b66fe/reference/CMakeLists.txt) par l'instruction:

```cmake
target_compile_options(ginkgo_reference PRIVATE "SHELL:-Wno-unused-command-line-argument -mllvm -polly -mllvm -polly-dependences-computeout=0 -mllvm -polly-vectorizer=stripmine -mllvm -polly-parallel")
```

**Notes**:

- Utiliser *LLVM/Polly* sur les implémentations optimisées pour *CUDA* ou *OpenMP* provoquent des erreurs à leurs exécutions (i.e. leurs compilations avec *LLVM/Polly* activé se déroulent comme un charme).
- Pour connaître les *SCoP* identifiés par *LLVM/Polly*, ajouter les options `-mllvm -polly-export` décrit comme:

  > Polly - Export Scops as JSON (Writes a .jscop file for each Scop)

  Mais cet option plante *Clang* pour ses versions 15.x: pour l'utiliser, il faut passer aux versions strictement supérieures à 15.x de *Clang*. De plus, cet option gère mal les noms des fichiers générés: il faut l'appliquer avec parcimonie...

- *LLVM/Polly* a l'avantage d'être indépendant du langage de programmation utilisé par le programmeur: *LLVM/Polly* travaille sur les *représentations intermédiaires* ou [*IR*](https://en.wikipedia.org/wiki/Intermediate_representation), générées par le compilateur, responsable de la validation de la syntaxe.
- Certaines fonctions impliquées dans le solveur *GC* de *Ginkgo* ne respectant pas les critères du *SCoP* (e.g. `step_1` et `step_2`), ne pouvant donc pas être optimisées par *LLVM/Polly* ont été reformulées: voire la section [PLUTO](#pluto) sur la fonction `step_1` pour plus de détail.
- *LLVM/Polly* n'est qu'un des projets de [*The LLVM Compiler Infrastructure*](https://llvm.org/).
- *LLVM/Polly* passe par *OpenMP* pour la parallélisation et la vectorisation.

#### PLUTO

*PLUTO* représente probablement l'état de l'art des *optimisations polyédriques*: *PLUTO* regroupe la majorité des outils développés, donc des recherches, sur le sujet. Ses caractéristiques:

- L'utilisateur doit baliser (avec les directives `#pragma scop` et `#pragma endscop`) son code pour instruire *PLUTO* des *SCoP*.
- *PLUTO* n'est pas intégré à un compilateur.
- *PLUTO* travaille sur le code source, ce qui le rend dépendant de la syntaxe du langage de programmation utilisé. Pour l'analyse syntaxique, *PLUTO* offre le choix entre [*Clan*](https://icps.u-strasbg.fr/~bastoul/development/clan/docs/clan.html) et [*Pet*](https://repo.or.cz/w/pet.git). *Pet* s'appuie sur *Clang* et offre un support complet du *C/C++* alors que *Clan* se limite à un *C* bridé. Cependant, l'utilisation de *Pet* par *PLUTO*, vers juin 2023, était impossible: ça ne fonctionnait tout simplement pas. **Note**: Alors que j'essayais d'améliorer l'intégration de *Pet* dans *PLUTO*, les auteurs de *PLUTO*, après des années d'inactivités sur leur [*github*](https://github.com/bondhugula/pluto), ont entrepris, entre autre, les mêmes travaux. [Une version stable](https://github.com/bondhugula/pluto/releases/download/0.12.0/pluto-0.12.0.tar.gz) de *PLUTO* est disponible depuis novembre 2023. Je ne l'ai pas, encore, testée.
- L'utilisation de *PLUTO* est un tantinet peu fastidieuse:

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

Constatant que la majorité des opérations impliquées dans le solveur *CG* de *Ginkgo* peut être optimisée si les matrices denses sont traitées par colonne plutôt que par ligne, une reformulation algorithmique a été effectuée. Tout en sachant que *Ginkgo* stockent ses matrices denses par ligne, et, donc, les parcourir par colonne dégraderont fortement les performances à cause des défauts de caches. Ces dégradations sont amplifiées lorsque les traitements sont parallélisés. **Note**: Pour éviter le problème de défaut de caches, il faut ajouter des opérations de transpositions coûteuses et en temps de calcul et en espace mémoire; une option inenvisageable lorsque le but est d'optimiser.

Lors de la reformulation, en exploitant les propriétés algébriques des opérations d'addition et de multiplication sur les éléments neutres (i.e. `0` et `1`), par, certes, l'ajout de tests et une complexité du code, mais qui peuvent éviter de lancer des calculs sur de potentiel vaste volume de données.

Par exemples:

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

  * `compute_norm2_dispatch` est `2,158910204` plus rapide,
  * `compute_conj_dot_dispatch` est `2,059145506` plus rapide,
  * `step_1` est `14,473490735` plus rapide,
  * `step_2` est `7,1691806` plus rapide

  que leurs versions séquentielles de référence.

- Les versions reformulées optimisées avec *OpenMP*:

  * `compute_norm2_dispatch` est `3,526641558` plus lente,
  * `compute_conj_dot_dispatch` est `2,508054004` plus lente,
  * `step_1` est `1,261263298` plus lente,
  * `step_2` est `1,459083835` plus lente

  que leurs versions optimisées avec *OpenMP* de référence.

En dépit des défauts de caches causés par les parcours des matrices denses par colonne alors qu'elles sont stockées par ligne.

**Notes**:

- Les versions séquentielles reformulées de `step_1` et `step_2` ne sont, respectivement, que `1,233304792` et `1,289617482` plus lentes que leurs versions optimisées par *OpenMP* de référence.
- Les concepteurs de *Ginkgo*, depuis 2021, songent à fournir plusieurs modes de stockage des matrices denses: [Clarify the behavioral differences between a dense matrix and a multivector](https://github.com/ginkgo-project/ginkgo/issues/796). Le débat est encore ouvert.

## Optimisations de OpenCARP

### Multi-GPU

### Task-based programming

### Algorithmiques du STL C++
