terraform {
  required_version = ">= 1.6"
  required_providers {
    helm = { source = "hashicorp/helm", version = "~> 2.17" }
  }
}
variable "kubeconfig" { type = string }
variable "image_repository" { type = string }
variable "existing_secret" { type = string }
variable "registry_file" { type = string }
provider "helm" {
  kubernetes { config_path = var.kubeconfig }
}
resource "helm_release" "inferscale" {
  name = "inferscale"
  namespace = "inferscale"
  create_namespace = true
  chart = "${path.module}/../helm/inferscale"
  set {
    name = "image.repository"
    value = var.image_repository
  }
  set {
    name = "existingSecret"
    value = var.existing_secret
  }
  values = [yamlencode({ registry = file(var.registry_file) })]
}
