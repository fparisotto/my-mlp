name := "myann"

version := "0.1"

scalaVersion := "2.12.4"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.2"

libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.2"

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
