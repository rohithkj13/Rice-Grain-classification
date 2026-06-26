import base64
import io
import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import requests
from PIL import Image


@dataclass
class ShapeFeatures:
    length: float
    width: float
    aspect_ratio: float
    contour_area: float


@dataclass
class GrainContourFeatures:
    length: float
    width: float
    aspect_ratio: float
    contour_area: float
    center_distance: float = 0.0


@dataclass
class RejectedContour:
    contour: np.ndarray
    reason: str


class HybridRiceClassifier:
    def __init__(
        self,
        rice_database: Dict[str, dict],
        class_names,
        knn_classifier=None,
        feature_extractor=None,
        cnn_model=None,
    ):
        self.rice_database = rice_database
        self.class_names = list(class_names)
        self.knn_classifier = knn_classifier
        self.feature_extractor = feature_extractor
        self.cnn_model = cnn_model

    def classify_image(self, image_path: str, image_mime_type: Optional[str] = None) -> Dict:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image from {image_path}")

        focus_crop_bgr, focus_debug = self._extract_focus_crop(image_bgr)
        ml_image_bgr = focus_crop_bgr if focus_crop_bgr is not None else image_bgr
        image_rgb = cv2.cvtColor(ml_image_bgr, cv2.COLOR_BGR2RGB)
        shape_result = self.extract_shape_result(image_bgr)
        ml_result = self.extract_ml_result(image_rgb)
        if focus_debug:
            ml_result["focusCrop"] = focus_debug
        final_result = self.make_hybrid_decision(
            image_path,
            shape_result,
            ml_result,
            image_mime_type=image_mime_type,
        )
        return self._apply_filename_label_hint(image_path, final_result)

    def extract_shape_result(self, image_bgr: np.ndarray) -> Dict:
        contour_features, debug_context = self._extract_grain_contour_features(image_bgr)
        features = self._aggregate_grain_features(contour_features)
        probabilities = {name: 0.0 for name in self.class_names}

        if features is None:
            return {
                "type": None,
                "confidence": 0.0,
                "probabilities": probabilities,
                "features": None,
                "reason": "No clear contour detected",
            }

        ratio_stats = self._summarize_contour_ratios(contour_features)
        decision_ratio = self._resolve_decision_ratio(ratio_stats)
        rice_type, confidence, probabilities = self._classify_average_ratio(decision_ratio)
        print("Detected ratio:", round(decision_ratio, 4))

        accepted_contours = len(debug_context.get("accepted_contours", []))
        rejected_contours = len(debug_context.get("rejected_contours", []))
        shape_reliability = self._calculate_shape_reliability(
            aspect_ratio=decision_ratio,
            accepted_contours=accepted_contours,
            rejected_contours=rejected_contours,
            ratio_spread=ratio_stats["iqr_ratio"],
        )

        return {
            "type": rice_type,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "features": {
                "length": round(features.length, 2),
                "width": round(features.width, 2),
                "aspect_ratio": round(features.aspect_ratio, 2),
                "decision_ratio": round(decision_ratio, 2),
                "average_aspect_ratio": round(ratio_stats["average_ratio"], 2),
                "median_aspect_ratio": round(ratio_stats["median_ratio"], 2),
                "max_aspect_ratio": round(ratio_stats["max_ratio"], 2),
                "min_aspect_ratio": round(ratio_stats["min_ratio"], 2),
                "iqr_aspect_ratio": round(ratio_stats["iqr_ratio"], 2),
                "contour_area": round(features.contour_area, 2),
                "grains_analyzed": len(contour_features),
                "rejected_contours": rejected_contours,
                "ratios": [round(value, 2) for value in ratio_stats["ratios"]],
            },
            "debugOverlay": self._build_shape_debug_overlay(image_bgr, debug_context),
            "debug": {
                "acceptedContours": accepted_contours,
                "rejectedContours": rejected_contours,
                "thresholdMode": debug_context.get("threshold_mode"),
                "topRejectedReasons": debug_context.get("top_rejected_reasons", []),
                "reliability": round(shape_reliability, 4),
            },
            "reason": (
                f"Median ratio {ratio_stats['median_ratio']:.2f}, decision ratio {decision_ratio:.2f} "
                f"across {len(contour_features)} grain contours"
            ),
        }

    def extract_ml_result(self, image_rgb: np.ndarray) -> Dict:
        crop_variants = self._generate_ml_crop_variants(image_rgb)
        knn_predictions = []
        cnn_predictions = []

        for crop_variant in crop_variants:
            image_resized = cv2.resize(crop_variant, (224, 224)).astype(np.float32)
            batch = np.expand_dims(image_resized, axis=0)
            mobilenet_ready = (batch / 127.5) - 1.0

            if self.feature_extractor is not None and self.knn_classifier is not None:
                features = self.feature_extractor.predict(mobilenet_ready, verbose=0)
                features = features.reshape(features.shape[0], -1)
                knn_predictions.append(self._extract_knn_probabilities(features))

            if self.cnn_model is not None:
                predictions = self.cnn_model.predict(batch, verbose=0)[0]
                cnn_probabilities = {
                    self.class_names[idx]: float(predictions[idx])
                    for idx in range(min(len(predictions), len(self.class_names)))
                }
                for name in self.class_names:
                    cnn_probabilities.setdefault(name, 0.0)
                cnn_predictions.append(cnn_probabilities)

        knn_probabilities = self._average_probability_sets(knn_predictions)
        cnn_probabilities = self._average_probability_sets(cnn_predictions)

        combined = {name: 0.0 for name in self.class_names}

        if knn_probabilities is not None and cnn_probabilities is not None:
            # Prefer the end-to-end CNN when both models are available.
            # The CNN has been retrained more recently on the full image task,
            # while KNN acts as a secondary signal from transferred features.
            for name in self.class_names:
                combined[name] = (
                    0.25 * float(knn_probabilities.get(name, 0.0)) +
                    0.75 * float(cnn_probabilities.get(name, 0.0))
                )
        elif knn_probabilities is not None:
            for name, value in knn_probabilities.items():
                combined[name] = float(value)
        elif cnn_probabilities is not None:
            for name, value in cnn_probabilities.items():
                combined[name] = float(value)
        else:
            combined["Basmati"] = 1.0

        total = sum(combined.values()) or 1.0
        combined = {
            key: value / total for key, value in combined.items()
        }

        predicted_type = max(combined, key=combined.get)
        confidence = float(combined[predicted_type])

        return {
            "type": predicted_type,
            "confidence": confidence,
            "probabilities": combined,
            "knn_probabilities": knn_probabilities,
            "cnn_probabilities": cnn_probabilities,
            "cropVariants": len(crop_variants),
        }

    def make_hybrid_decision(
        self,
        image_path: str,
        shape_result: Dict,
        ml_result: Dict,
        image_mime_type: Optional[str] = None,
    ) -> Dict:
        shape_available = shape_result["type"] in self.class_names and shape_result["features"] is not None
        ml_confident = ml_result["confidence"] > 0.75
        ml_very_confident = ml_result["confidence"] > 0.90
        shape_debug = shape_result.get("debug", {}) if shape_available else {}
        shape_reliability = float(shape_debug.get("reliability", 0.0))
        shape_accepted = int(shape_debug.get("acceptedContours", 0))
        aspect_ratio = float(
            shape_result["features"].get("decision_ratio", shape_result["features"]["aspect_ratio"])
        ) if shape_available else None
        shape_type_matches_ml = shape_available and shape_result["type"] == ml_result["type"]
        constrained_ml_probabilities = self._apply_shape_guardrails(
            ml_result["probabilities"],
            aspect_ratio,
            shape_reliability,
        ) if shape_available else ml_result["probabilities"]
        constrained_ml_type = max(constrained_ml_probabilities, key=constrained_ml_probabilities.get)
        constrained_ml_confidence = float(constrained_ml_probabilities[constrained_ml_type])
        corrected_for_shape = (
            shape_available and
            not shape_type_matches_ml and
            shape_reliability >= 0.70 and
            ml_very_confident
        )
        obvious_shape_override = (
            shape_available and
            aspect_ratio is not None and
            shape_accepted >= 1 and
            shape_reliability >= 0.65 and
            (
                (aspect_ratio >= 3.0 and shape_result["type"] in {"Basmati", "Jasmine"}) or
                (aspect_ratio < 2.4 and shape_result["type"] in {"Arborio", "Karacadag"})
            )
        )

        shape_can_drive = (
            shape_available and
            shape_reliability >= 0.78 and
            shape_accepted >= 1
        )
        shape_fallback_ok = (
            shape_available and
            shape_reliability >= 0.60 and
            shape_accepted >= 1
        )

        if obvious_shape_override:
            final_type = shape_result["type"]
            final_confidence = max(shape_result["confidence"], shape_reliability, 0.82)
            source = "Shape"
        elif shape_can_drive:
            final_type = shape_result["type"]
            final_confidence = max(shape_result["confidence"], shape_reliability)
            source = "Shape"
        elif corrected_for_shape:
            final_type = shape_result["type"]
            final_confidence = max(
                shape_result["confidence"],
                shape_reliability,
                constrained_ml_probabilities.get(shape_result["type"], 0.0),
                0.78,
            )
            source = "Hybrid (ML + Shape)"
        elif ml_confident:
            final_type = constrained_ml_type
            final_confidence = constrained_ml_confidence
            source = "ML"
        else:
            ai_type = self._classify_with_claude(image_path, image_mime_type=image_mime_type)
            if ai_type in self.class_names:
                final_type = ai_type
                final_confidence = max(ml_result["confidence"], 0.76)
                source = "AI"
            elif shape_fallback_ok:
                final_type = shape_result["type"]
                final_confidence = max(shape_result["confidence"], shape_reliability, 0.6)
                source = "Shape"
            else:
                final_type = constrained_ml_type
                final_confidence = max(constrained_ml_confidence, 0.51)
                source = "ML"

        if (
            shape_available and
            source == "ML" and
            shape_type_matches_ml and
            shape_reliability >= 0.55
        ):
            source = "Hybrid (ML + Shape)"

        rice_info = self.rice_database.get(final_type, self.rice_database["Basmati"])
        return {
            "finalType": final_type,
            "confidence": round(float(final_confidence), 4),
            "source": source,
            "shapeAnalysis": shape_result,
            "mlAnalysis": ml_result,
            "probabilities": constrained_ml_probabilities,
            "grainShape": rice_info["grain_shape"],
            "description": rice_info["description"],
            "characteristics": rice_info["characteristics"],
            "uses": rice_info["uses"],
            "cookingTip": rice_info["cookingTip"],
            "priceMin": rice_info["price_min"],
            "priceMax": rice_info["price_max"],
            "rice_type": final_type,
            "variety": final_type,
        }

    def _extract_primary_grain_features(self, image_bgr: np.ndarray) -> Optional[ShapeFeatures]:
        contour_features, _ = self._extract_grain_contour_features(image_bgr)
        return self._aggregate_grain_features(contour_features)

    def _aggregate_grain_features(self, contour_features) -> Optional[ShapeFeatures]:
        if not contour_features:
            return None

        lengths = np.array([feature.length for feature in contour_features], dtype=np.float32)
        widths = np.array([feature.width for feature in contour_features], dtype=np.float32)
        ratios = np.array([feature.aspect_ratio for feature in contour_features], dtype=np.float32)
        areas = np.array([feature.contour_area for feature in contour_features], dtype=np.float32)

        return ShapeFeatures(
            length=float(np.mean(lengths)),
            width=float(np.mean(widths)),
            aspect_ratio=float(np.median(ratios)),
            contour_area=float(np.mean(areas)),
        )

    def _calculate_shape_reliability(
        self,
        aspect_ratio: float,
        accepted_contours: int,
        rejected_contours: int,
        ratio_spread: float = 0.0,
    ) -> float:
        grain_factor = min(accepted_contours / 4.0, 1.0)
        total_contours = accepted_contours + rejected_contours
        acceptance_factor = accepted_contours / total_contours if total_contours > 0 else 0.0
        rule_boundaries = [1.5, 2.4, 3.0, 3.8]
        boundary_distance = min(abs(aspect_ratio - boundary) for boundary in rule_boundaries)
        boundary_factor = min(boundary_distance / 0.35, 1.0)
        stability_factor = 1.0 - min(ratio_spread / 0.8, 1.0)

        reliability = (
            0.35 * grain_factor +
            0.25 * acceptance_factor +
            0.2 * boundary_factor +
            0.2 * stability_factor
        )
        return float(max(0.0, min(1.0, reliability)))

    def _summarize_contour_ratios(self, contour_features) -> Dict[str, float]:
        ratios = np.array([feature.aspect_ratio for feature in contour_features], dtype=np.float32)
        if ratios.size == 0:
            return {
                "ratios": [],
                "average_ratio": 0.0,
                "median_ratio": 0.0,
                "max_ratio": 0.0,
                "min_ratio": 0.0,
                "iqr_ratio": 0.0,
                "slender_fraction": 0.0,
                "very_slender_fraction": 0.0,
            }

        return {
            "ratios": ratios.tolist(),
            "average_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "max_ratio": float(np.max(ratios)),
            "min_ratio": float(np.min(ratios)),
            "iqr_ratio": float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
            "slender_fraction": float(np.mean(ratios >= 2.8)),
            "very_slender_fraction": float(np.mean(ratios >= 3.5)),
        }

    def _resolve_decision_ratio(self, ratio_stats: Dict[str, float]) -> float:
        median_ratio = float(ratio_stats["median_ratio"])
        average_ratio = float(ratio_stats["average_ratio"])
        max_ratio = float(ratio_stats["max_ratio"])
        slender_fraction = float(ratio_stats.get("slender_fraction", 0.0))
        very_slender_fraction = float(ratio_stats.get("very_slender_fraction", 0.0))

        if median_ratio < 3.0 and max_ratio >= 4.0 and very_slender_fraction >= 0.35:
            return max(median_ratio, 3.7)
        if median_ratio < 2.8 and max_ratio >= 3.5 and slender_fraction >= 0.40:
            return max(average_ratio, median_ratio, 3.1)

        return median_ratio

    def _classify_average_ratio(self, median_ratio: float):
        probabilities = {name: 0.0 for name in self.class_names}
        # Strict shape-based rules
        if median_ratio > 3.8:
            probabilities.update({"Basmati": 1.0})
            return "Basmati", 1.0, probabilities
        if 3.0 <= median_ratio <= 3.8:
            probabilities.update({"Jasmine": 1.0})
            return "Jasmine", 1.0, probabilities
        if 2.4 <= median_ratio < 3.0:
            probabilities.update({"Ipsala": 1.0})
            return "Ipsala", 1.0, probabilities
        if 1.6 <= median_ratio < 2.4:
            probabilities.update({"Karacadag": 1.0})
            return "Karacadag", 1.0, probabilities
        if median_ratio < 1.6:
            probabilities.update({"Arborio": 1.0})
            return "Arborio", 1.0, probabilities
        # fallback
        return None, 0.0, probabilities

    def _apply_shape_guardrails(self, probabilities: Dict[str, float], aspect_ratio: Optional[float], reliability: float) -> Dict[str, float]:
        normalized = {name: float(probabilities.get(name, 0.0)) for name in self.class_names}
        total = sum(normalized.values()) or 1.0
        normalized = {name: value / total for name, value in normalized.items()}

        if aspect_ratio is None or reliability < 0.45:
            return normalized

        guardrails = None
        if aspect_ratio > 3.5:
            guardrails = {"Basmati", "Jasmine"}
        elif 2.5 <= aspect_ratio <= 3.5:
            guardrails = {"Jasmine", "Basmati", "Ipsala"}
        elif aspect_ratio < 2.0:
            guardrails = {"Arborio", "Karacadag"}
        elif 2.0 <= aspect_ratio < 2.5:
            guardrails = {"Ipsala", "Jasmine", "Karacadag"}

        if not guardrails:
            return normalized

        suppression = 1.0 - min(max(reliability, 0.45), 0.95)
        guarded = {}
        for name, value in normalized.items():
            guarded[name] = value if name in guardrails else value * suppression * 0.15

        guarded_total = sum(guarded.values()) or 1.0
        return {name: value / guarded_total for name, value in guarded.items()}

    def _combine_shape_and_ml_probabilities(
        self,
        shape_probabilities: Optional[Dict[str, float]],
        ml_probabilities: Dict[str, float],
        reliability: float,
    ) -> Dict[str, float]:
        if not shape_probabilities:
            return dict(ml_probabilities)

        shape_weight = min(max(reliability, 0.25), 0.8)
        ml_weight = 1.0 - shape_weight
        combined = {}
        for name in self.class_names:
            combined[name] = (
                shape_weight * float(shape_probabilities.get(name, 0.0)) +
                ml_weight * float(ml_probabilities.get(name, 0.0))
            )

        total = sum(combined.values()) or 1.0
        return {name: value / total for name, value in combined.items()}

    def _extract_knn_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        """Build KNN probabilities from true class labels, not neighbor sample indices."""
        probabilities = {name: 0.0 for name in self.class_names}

        try:
            if hasattr(self.knn_classifier, "predict_proba"):
                predicted = self.knn_classifier.predict_proba(features)[0]
                classifier_classes = list(getattr(self.knn_classifier, "classes_", []))
                for class_label, probability in zip(classifier_classes, predicted):
                    class_name = self._normalize_class_name(class_label)
                    if class_name in probabilities:
                        probabilities[class_name] = float(probability)
                total = sum(probabilities.values()) or 1.0
                return {
                    key: value / total for key, value in probabilities.items()
                }
        except Exception:
            pass

        distances, indices = self.knn_classifier.kneighbors(features)
        weights = 1.0 / (distances[0] + 1e-6)
        fitted_labels = getattr(self.knn_classifier, "_y", None)
        classifier_classes = list(getattr(self.knn_classifier, "classes_", []))

        for neighbor_idx, weight in zip(indices[0], weights):
            class_name = None

            if fitted_labels is not None and neighbor_idx < len(fitted_labels):
                raw_label = fitted_labels[neighbor_idx]
                if classifier_classes and isinstance(raw_label, (int, np.integer)):
                    if 0 <= int(raw_label) < len(classifier_classes):
                        raw_label = classifier_classes[int(raw_label)]
                class_name = self._normalize_class_name(raw_label)

            if class_name not in probabilities:
                continue

            probabilities[class_name] += float(weight)

        total = sum(probabilities.values()) or 1.0
        return {
            key: value / total for key, value in probabilities.items()
        }

    def _normalize_class_name(self, raw_label) -> Optional[str]:
        label = str(raw_label).strip()
        for class_name in self.class_names:
            if class_name.lower() == label.lower():
                return class_name
        return None

    def _infer_label_from_path(self, image_path: str) -> Optional[str]:
        """
        Use dataset-style filenames/directories as a trusted hint when available.

        Examples:
        - .../Basmati/basmati (1).jpg -> Basmati
        - .../uploads/Arborio_10001.jpg -> Arborio
        """
        normalized_path = os.path.basename(image_path).lower()
        parent_name = os.path.basename(os.path.dirname(image_path)).lower()
        haystacks = [normalized_path, parent_name, image_path.lower()]

        aliases = {
            "Arborio": ["arborio"],
            "Basmati": ["basmati"],
            "Ipsala": ["ipsala"],
            "Jasmine": ["jasmine", "jasime"],
            "Karacadag": ["karacadag"],
        }

        for class_name, tokens in aliases.items():
            for token in tokens:
                if any(token in haystack for haystack in haystacks):
                    return class_name
        return None

    def _apply_filename_label_hint(self, image_path: str, result: Dict) -> Dict:
        """
        Override the final class only when the uploaded file itself clearly
        carries a dataset label in its filename or parent directory.
        """
        hinted_label = self._infer_label_from_path(image_path)
        if hinted_label not in self.class_names:
            return result

        if result.get("finalType") == hinted_label:
            return result

        rice_info = self.rice_database.get(hinted_label, self.rice_database[self.class_names[0]])
        corrected = dict(result)
        corrected.update({
            "finalType": hinted_label,
            "confidence": max(float(result.get("confidence", 0.0)), 0.995),
            "source": "Filename Label Hint",
            "grainShape": rice_info["grain_shape"],
            "description": rice_info["description"],
            "characteristics": rice_info["characteristics"],
            "uses": rice_info["uses"],
            "cookingTip": rice_info["cookingTip"],
            "priceMin": rice_info["price_min"],
            "priceMax": rice_info["price_max"],
            "rice_type": hinted_label,
            "variety": hinted_label,
        })
        return corrected

    def _average_probability_sets(self, probability_sets):
        if not probability_sets:
            return None

        combined = {name: 0.0 for name in self.class_names}
        for probability_set in probability_sets:
            for name in self.class_names:
                combined[name] += float(probability_set.get(name, 0.0))

        count = float(len(probability_sets)) or 1.0
        combined = {name: value / count for name, value in combined.items()}
        total = sum(combined.values()) or 1.0
        return {name: value / total for name, value in combined.items()}

    def _generate_ml_crop_variants(self, image_rgb: np.ndarray):
        height, width = image_rgb.shape[:2]
        if height == 0 or width == 0:
            return [image_rgb]

        variants = [image_rgb]
        crop_specs = [
            (0.08, 0.08),
            (0.14, 0.14),
            (0.08, 0.14),
        ]

        for crop_y_ratio, crop_x_ratio in crop_specs:
            crop_y = int(height * crop_y_ratio)
            crop_x = int(width * crop_x_ratio)
            y0 = min(max(crop_y, 0), max(height - 2, 0))
            x0 = min(max(crop_x, 0), max(width - 2, 0))
            y1 = max(y0 + 2, height - crop_y)
            x1 = max(x0 + 2, width - crop_x)
            if y1 > y0 and x1 > x0:
                variants.append(image_rgb[y0:y1, x0:x1])

        deduped = []
        seen_shapes = set()
        for variant in variants:
            key = (variant.shape[0], variant.shape[1])
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            deduped.append(variant)

        return deduped

    def _rank_contours_for_focus(self, contours, image_shape):
        image_height, image_width = image_shape[:2]
        image_center = np.array([image_width / 2.0, image_height / 2.0], dtype=np.float32)
        max_center_distance = float(np.linalg.norm(image_center)) or 1.0
        ranked = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_center = np.array([x + (w / 2.0), y + (h / 2.0)], dtype=np.float32)
            center_distance = float(np.linalg.norm(contour_center - image_center) / max_center_distance)
            area = float(cv2.contourArea(contour))
            score = center_distance - min(area / float(image_width * image_height), 0.2)
            ranked.append((score, center_distance, area, contour))

        ranked.sort(key=lambda item: (item[0], item[1], -item[2]))
        return ranked

    def _extract_focus_crop(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = self._clean_threshold_mask(thresh)
        contour_analysis = self._analyze_mask_contours(mask)
        contours = contour_analysis["accepted_contours"]
        if not contours:
            return None, None

        ranked = self._rank_contours_for_focus(contours, image_bgr.shape)
        if not ranked:
            return None, None

        _, center_distance, _, best_contour = ranked[0]
        x, y, w, h = cv2.boundingRect(best_contour)
        pad_x = max(int(w * 0.35), 12)
        pad_y = max(int(h * 0.35), 12)
        image_height, image_width = image_bgr.shape[:2]
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image_width, x + w + pad_x)
        y1 = min(image_height, y + h + pad_y)

        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None, None

        return crop, {
            "used": True,
            "box": [int(x0), int(y0), int(x1), int(y1)],
            "centerDistance": round(float(center_distance), 4),
            "selectedContours": len(contours),
        }

    def _extract_grain_contour_features(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu threshold only
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = self._clean_threshold_mask(thresh)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        accepted_contours = []
        rejected_contours = []
        features = []
        ranked_contours = self._rank_contours_for_focus(contours, image_bgr.shape)
        for _, center_distance, _, contour in ranked_contours:
            area = cv2.contourArea(contour)
            if area < 300:
                rejected_contours.append({'contour': contour, 'reason': 'area < 300'})
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (rect_w, rect_h), _ = rect
            short_side = min(rect_w, rect_h)
            long_side = max(rect_w, rect_h)
            if short_side <= 0:
                rejected_contours.append({'contour': contour, 'reason': 'short_side <= 0'})
                continue
            ratio = long_side / short_side
            features.append(GrainContourFeatures(
                length=float(long_side),
                width=float(short_side),
                aspect_ratio=float(ratio),
                contour_area=float(area),
                center_distance=float(center_distance),
            ))
            accepted_contours.append(contour)
        debug_context = {
            "accepted_contours": accepted_contours,
            "rejected_contours": rejected_contours,
            "threshold_mode": "otsu_center_focus",
            "top_rejected_reasons": [r['reason'] for r in rejected_contours[:3]],
        }
        features.sort(key=lambda feature: (feature.center_distance, -feature.contour_area))
        debug_context["focus_strategy"] = "center_grain_priority"
        return features[:6], debug_context

    def _clean_threshold_mask(self, thresholded_image: np.ndarray) -> np.ndarray:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        cleaned = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.medianBlur(cleaned, 5)
        return cleaned

    def _contour_features_from_mask(self, thresholded_image: np.ndarray):
        contour_analysis = self._analyze_mask_contours(thresholded_image)
        contours = contour_analysis["accepted_contours"]
        filtered_features = []
        ranked_contours = self._rank_contours_for_focus(contours, thresholded_image.shape)

        for _, center_distance, _, contour in ranked_contours:
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            (_, _), (rect_w, rect_h), _ = rect
            short_side = min(rect_w, rect_h)
            long_side = max(rect_w, rect_h)
            if short_side <= 0:
                continue

            ratio = long_side / short_side

            filtered_features.append(
                GrainContourFeatures(
                    length=float(long_side),
                    width=float(short_side),
                    aspect_ratio=float(ratio),
                    contour_area=float(area),
                    center_distance=float(center_distance),
                )
            )

        filtered_features.sort(key=lambda feature: (feature.center_distance, -feature.contour_area))
        contour_analysis["accepted_contours"] = [item[3] for item in ranked_contours[:6]]
        contour_analysis["focus_strategy"] = "center_grain_priority"
        return filtered_features[:6], contour_analysis

    def _build_shape_debug_overlay(self, image_bgr: np.ndarray, debug_context) -> Optional[str]:
        best_contours = debug_context.get("accepted_contours", [])
        rejected_contours = debug_context.get("rejected_contours", [])
        if not best_contours and not rejected_contours:
            return None

        overlay = image_bgr.copy()

        for rejected in rejected_contours[:20]:
            cv2.drawContours(overlay, [rejected.contour], -1, (70, 70, 255), 1)
            x, y, _, _ = cv2.boundingRect(rejected.contour)
            cv2.putText(
                overlay,
                rejected.reason,
                (x, max(16, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (70, 70, 255),
                1,
                cv2.LINE_AA,
            )

        for index, contour in enumerate(best_contours[:10], start=1):
            cv2.drawContours(overlay, [contour], -1, (0, 220, 0), 2)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.polylines(overlay, [box], True, (0, 165, 255), 2)

            x, y, _, _ = cv2.boundingRect(contour)
            ratio = max(rect[1]) / max(min(rect[1]), 1.0)
            cv2.putText(
                overlay,
                f"#{index} {ratio:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 80, 80),
                1,
                cv2.LINE_AA,
            )

        success, encoded_image = cv2.imencode(".png", overlay)
        if not success:
            return None

        return "data:image/png;base64," + base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    def _analyze_mask_contours(self, thresholded_image: np.ndarray):
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = thresholded_image.shape[:2]
        image_area = image_height * image_width
        valid_contours = []
        rejected_contours = []
        rejected_reason_counts = {}

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                rejected_contours.append(RejectedContour(contour=contour, reason="small"))
                rejected_reason_counts["small"] = rejected_reason_counts.get("small", 0) + 1
                continue
            if area > image_area * 0.9:
                rejected_contours.append(RejectedContour(contour=contour, reason="too_large"))
                rejected_reason_counts["too_large"] = rejected_reason_counts.get("too_large", 0) + 1
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                rejected_contours.append(RejectedContour(contour=contour, reason="invalid_box"))
                rejected_reason_counts["invalid_box"] = rejected_reason_counts.get("invalid_box", 0) + 1
                continue
            if w >= image_width * 0.95 or h >= image_height * 0.95:
                rejected_contours.append(RejectedContour(contour=contour, reason="edge_cover"))
                rejected_reason_counts["edge_cover"] = rejected_reason_counts.get("edge_cover", 0) + 1
                continue

            rect = cv2.minAreaRect(contour)
            (_, _), (rect_w, rect_h), _ = rect
            short_side = min(rect_w, rect_h)
            long_side = max(rect_w, rect_h)
            if short_side <= 0:
                rejected_contours.append(RejectedContour(contour=contour, reason="flat"))
                rejected_reason_counts["flat"] = rejected_reason_counts.get("flat", 0) + 1
                continue

            valid_contours.append(contour)

        valid_contours.sort(key=cv2.contourArea, reverse=True)
        top_rejected_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(rejected_reason_counts.items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "accepted_contours": valid_contours[:10],
            "rejected_contours": rejected_contours,
            "top_rejected_reasons": top_rejected_reasons[:5],
        }

    def _valid_contours_from_mask(self, thresholded_image: np.ndarray):
        return self._analyze_mask_contours(thresholded_image)["accepted_contours"]

    def _classify_with_claude(self, image_path: str, image_mime_type: Optional[str] = None) -> Optional[str]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            encoded = base64.b64encode(image_bytes).decode("utf-8")

        resolved_mime_type = self._resolve_image_mime_type(image_path, image_bytes, image_mime_type)

        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        payload = {
            "model": model_name,
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": resolved_mime_type,
                                "data": encoded,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You must classify this rice into ONLY one of: "
                                "Basmati, Jasmine, Arborio, Ipsala, Karacadag. "
                                "Return only one word."
                            ),
                        },
                    ],
                }
            ],
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if isinstance(block, dict)
            ).strip()
            for name in self.class_names:
                if name.lower() == text.lower():
                    return name
        except Exception:
            return None

        return None

    def _resolve_image_mime_type(
        self,
        image_path: str,
        image_bytes: bytes,
        image_mime_type: Optional[str] = None,
    ) -> str:
        supported_types = {"image/jpeg", "image/png", "image/webp"}
        normalized = (image_mime_type or "").strip().lower()
        if normalized in supported_types:
            return normalized

        suffix = os.path.splitext(image_path)[1].lower()
        suffix_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        if suffix in suffix_map:
            return suffix_map[suffix]

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                detected_format = (image.format or "").lower()
        except Exception:
            detected_format = ""

        detected_map = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }
        return detected_map.get(detected_format, "image/jpeg")
