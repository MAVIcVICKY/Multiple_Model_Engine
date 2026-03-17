from django.core.management.base import BaseCommand
from core.pinecone_db import _get_index

class Command(BaseCommand):
    help = "Show current status of the Pinecone index"

    def handle(self, *args, **options):
        try:
            index = _get_index()
            stats = index.describe_index_stats()
            
            self.stdout.write(self.style.SUCCESS("\n📊 Pinecone Index Stats:"))
            self.stdout.write(f"   Total Vectors: {stats['total_vector_count']}")
            self.stdout.write(f"   Storage Fullness: {stats.get('storageFullness', 0)}")
            self.stdout.write("-" * 30)
            
            self.stdout.write(self.style.NOTICE("\nNamespace Details:"))
            for ns, data in stats.get('namespaces', {}).items():
                self.stdout.write(f"   Namespace: '{ns}' - Count: {data['vector_count']}")
            
            self.stdout.write(self.style.SUCCESS("\n✅ Successfully retrieved stats.\n"))
            
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"❌ Error: {str(e)}"))
