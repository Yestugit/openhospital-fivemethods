/* Lifecycle-only launcher for the official Neo4j Community entry point.
 * A local stop file triggers System.exit so Neo4j's normal JVM shutdown hooks
 * flush and close the database. No Windows service or visible console needed.
 */
import java.nio.file.Files;
import java.nio.file.Path;
import org.neo4j.server.CommunityEntryPoint;

class LocalMemOSNeo4j {
    public static void main(String[] args) throws Exception {
        String directory = System.getenv("MEMATLAS_NEO4J_STOP_FILE");
        if (directory == null) throw new IllegalStateException("Missing local stop file");
        Path signal = Path.of(directory);
        Thread watcher = new Thread(() -> {
            try {
                while (!Files.exists(signal)) Thread.sleep(250);
                Files.delete(signal);
                System.exit(0);
            } catch (Exception error) {
                System.err.println("Local Neo4j stop watcher failed: " + error.getClass().getName());
            }
        }, "mematlas-local-stop");
        watcher.setDaemon(true);
        watcher.start();
        CommunityEntryPoint.main(args);
    }
}
