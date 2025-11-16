#!/usr/bin/env python3
"""
Script para analizar videos de baloncesto
"""

import argparse
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from main import BasketballAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analizar videos de baloncesto"
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Ruta al video de entrada'
    )

    parser.add_argument(
        '--output',
        help='Ruta para guardar video procesado'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Archivo de configuración'
    )

    parser.add_argument(
        '--save-results',
        help='Directorio para guardar resultados'
    )

    parser.add_argument(
        '--no-heatmap',
        action='store_true',
        help='No generar heatmaps'
    )

    parser.add_argument(
        '--no-classification',
        action='store_true',
        help='No clasificar jugadas'
    )

    parser.add_argument(
        '--generate-highlights',
        action='store_true',
        help='Generar video de highlights'
    )

    args = parser.parse_args()

    # Verificar que el video existe
    if not Path(args.input).exists():
        print(f"Error: Video no encontrado: {args.input}")
        sys.exit(1)

    # Inicializar analizador
    print("Inicializando analizador...")
    analyzer = BasketballAnalyzer(config_path=args.config)

    # Procesar video
    print(f"Procesando video: {args.input}")
    results = analyzer.process_video(
        video_path=args.input,
        output_path=args.output,
        generate_heatmap=not args.no_heatmap,
        classify_plays=not args.no_classification,
        show_progress=True
    )

    # Mostrar estadísticas
    print("\n" + "="*60)
    print("RESULTADOS DEL ANÁLISIS")
    print("="*60)
    print(analyzer.stats_calc.format_report(results['statistics']))

    # Guardar resultados
    if args.save_results:
        print(f"\nGuardando resultados en: {args.save_results}")
        analyzer.save_results(results, args.save_results)

    # Generar highlights
    if args.generate_highlights and args.output:
        print("\nGenerando highlights...")
        highlights_path = args.output.replace('.mp4', '_highlights.mp4')
        analyzer.generate_highlights(
            args.input,
            results,
            highlights_path
        )

    print("\n¡Análisis completado!")


if __name__ == "__main__":
    main()
