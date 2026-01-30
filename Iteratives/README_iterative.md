# 🔄 Sistema de Reconstrução Iterativa Unificado

## 📋 Visão Geral

Este sistema consolidou os dois scripts MAIN anteriores (`MAIN_TIGRE_iterative_basic.py` e `MAIN_TIGRE_iterative_TV.py`) num único script unificado com configuração centralizada de parâmetros.

## 📁 Ficheiros

- **`MAIN_TIGRE_iterative.py`** - Script principal unificado
- **`iterative_parameters.py`** - Configurações de todos os algoritmos com explicações detalhadas

## 🚀 Como Usar

### Método 1: Escolher Algoritmo Diretamente (Recomendado)

1. Abra `MAIN_TIGRE_iterative.py`
2. Na secção "ALGORITHM SELECTION" (linha ~395), descomente o algoritmo desejado:

```python
# 📊 Algoritmos BÁSICOS:
algorithm_config = get_algorithm_config('SIRT')       # Clássico
#algorithm_config = get_algorithm_config('CGLS')      # Rápido
#algorithm_config = get_algorithm_config('OSSART')    # Preview rápido

# 🎯 Algoritmos com TV (remove artefactos):
#algorithm_config = get_algorithm_config('OSSART_TV') # Melhor para metal
#algorithm_config = get_algorithm_config('SART_TV')   # Mais preciso
```

3. Configure o dataset na secção "SELECT YOUR DATASET" (linha ~470)
4. Execute!

### Método 2: Usar Preset Otimizado

Use configurações pré-definidas para casos comuns:

```python
# Descomente um preset:
algorithm_config = get_preset_config('fantoma_pmma')      # PMMA
#algorithm_config = get_preset_config('metal_artifacts')  # Artefactos metal
#algorithm_config = get_preset_config('preview_rapido')   # Preview rápido
```

## 🎯 Algoritmos Disponíveis

### Algoritmos Básicos (sem TV)
- **SIRT** - Clássico, equilibrado qualidade/velocidade
- **CGLS** - Rápido, bom para detalhes
- **LSQR/LSMR** - Numericamente estável
- **OSSART** - Muito rápido (preview)
- **SART** - Alternativa ao SIRT

### Algoritmos TV (remove artefactos)
- **OSSART_TV** - ⭐ RECOMENDADO para artefactos de metal
- **SART_TV** - Mais preciso que OSSART_TV
- **ASD_POCS** - Artefactos severos
- **AWASD_POCS** - Versão adaptativa

## 📊 Presets Disponíveis

- `fantoma_pmma` - Otimizado para fantomas PMMA
- `fantoma_agua` - Otimizado para água
- `padrao_barras` - Teste de resolução
- `metal_artifacts` - Remove artefactos de metal
- `ruido_alto` - Dados ruidosos
- `preview_rapido` - Preview rápido
- `maxima_qualidade` - Máxima qualidade (lento)

## 🔧 Personalização de Parâmetros

Se precisar ajustar parâmetros específicos:

```python
# Escolher algoritmo
algorithm_config = get_algorithm_config('OSSART_TV')

# Personalizar parâmetros
algorithm_config['iterations'] = 150  # Aumentar iterações
algorithm_config['params']['tv_lambda'] = 30.0  # Mais suavização
algorithm_config['params']['blocksize'] = 30  # Ajustar blocksize
```

## 📖 Guia de Parâmetros

### ITERATIONS (Iterações)
- **↑ Aumentar**: Melhor qualidade, mais tempo
- **↓ Diminuir**: Mais rápido, pode perder qualidade
- **Valores típicos**: 30-200

### TV_LAMBDA (Regularização TV)
- **↑ Aumentar**: Remove mais artefactos/ruído, perde detalhes
- **↓ Diminuir**: Preserva detalhes, mais artefactos
- **Valores típicos**: 5-50

### BLOCKSIZE (OSSART apenas)
- **↑ Aumentar**: Mais rápido, menos preciso
- **↓ Diminuir**: Mais lento, mais preciso
- **Valores típicos**: 10-40

### TV_NG (Iterações TV internas)
- **Valor recomendado**: 20-25 (raramente precisa ajustar)

### ASD_ALPHA (ASD_POCS apenas)
- **↑ Aumentar**: Convergência rápida, pode instabilizar
- **↓ Diminuir**: Mais estável, mais lento
- **Valores típicos**: 0.001-0.005

### ASD_EPSILON (ASD_POCS apenas)
- **↑ Aumentar**: Mais suavização
- **↓ Diminuir**: Mais fiel aos dados
- **Valores típicos**: 0.01-0.1

## 💡 Dicas de Uso

### Para ver informação sobre algoritmos:

```python
# Ver todos os algoritmos disponíveis
list_available_algorithms()

# Ver todos os presets
list_presets()

# Ver info detalhada de um algoritmo
print_algorithm_info('OSSART_TV')
```

### Quando usar cada algoritmo:

| Caso | Algoritmo Recomendado |
|------|----------------------|
| Fantoma PMMA simples | SIRT ou CGLS |
| Fantoma água | CGLS |
| Padrão de barras | CGLS ou OSSART |
| Artefactos de metal | OSSART_TV ⭐ |
| Dados ruidosos | SART_TV |
| Preview rápido | OSSART (blocksize alto) |
| Máxima qualidade | SIRT (muitas iterações) |

## 🔍 Workflow Típico

1. **Escolher algoritmo** apropriado para o seu caso
2. **Configurar geometria** (voxel size, DSD, DSO, etc.)
3. **Selecionar dataset**
4. **Executar primeira reconstrução**
5. **Avaliar resultado**:
   - Muito ruído? → Aumentar `iterations` ou usar TV
   - Perdendo detalhes? → Diminuir `tv_lambda`
   - Muito lento? → Usar OSSART ou aumentar `blocksize`
6. **Ajustar parâmetros** se necessário
7. **Reconstruir novamente**

## 📝 Vantagens do Novo Sistema

✅ **Um único script** em vez de dois  
✅ **Parâmetros centralizados** e documentados  
✅ **Explicações detalhadas** de cada parâmetro  
✅ **Presets otimizados** para casos comuns  
✅ **Fácil personalização** quando necessário  
✅ **Código mais limpo** e mantível  

## ⚠️ Notas Importantes

- Os scripts antigos (`MAIN_TIGRE_iterative_basic.py` e `MAIN_TIGRE_iterative_TV.py`) podem ser mantidos como backup, mas o novo `MAIN_TIGRE_iterative.py` substitui ambos
- Todas as explicações detalhadas dos parâmetros estão em `iterative_parameters.py`
- Para adicionar novos algoritmos no futuro, basta editar `iterative_parameters.py`

## 🆘 Troubleshooting

**Problema**: "Unknown algorithm"  
**Solução**: Verifique o nome do algoritmo em `list_available_algorithms()`

**Problema**: Reconstrução muito lenta  
**Solução**: Use OSSART com blocksize alto, ou reduza iterações

**Problema**: Muitos artefactos de metal  
**Solução**: Use `OSSART_TV` ou preset `metal_artifacts`

**Problema**: Perdendo detalhes finos  
**Solução**: Diminua `tv_lambda` ou use algoritmo básico sem TV
